"""One pipeline model with four independently trained RAU-adapted experts."""

import math
import pathlib
import time

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim

from common.base_model import BaseModel
from common import settings, utils
from .river_attention import RiverUNetPlusPlus


IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_DIR = pathlib.Path(__file__).resolve().parent
ARCHITECTURE = "effb3-unetpp-raff-cbam-four-binary-experts-v1"
EXPERT_ARCHITECTURE = "effb3-unetpp-raff-cbam-binary-expert-v1"
FUSION_POLICY = "supported-crossing-then-max-positive-artery-vein-class4-v1"
EXPERT_SPECS = {
    "arteries_crossings": (1, 3),
    "veins_crossings": (2, 3),
    "crossings": (3,),
    "class4": (4,),
}

train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]
val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class WeightedCEDiceLoss(nn.Module):
    """The baseline loss applied to each expert's rest/positive logits."""

    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def _snapshot_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return value


class UNetPlusPlusEffB3RiverEnsemble(BaseModel):
    """Four networks, one fit/predict/save/load contract and one final label map.

    Only the active expert occupies the GPU. All idle networks and optimizer
    snapshots reside on CPU, including when the runner creates three folds.
    """

    def __init__(self, encoder_weights="imagenet", thresholds=None):
        super().__init__()
        if settings.NUM_CLASSES != 5:
            raise ValueError("This ensemble requires the original five-class masks")
        self.num_classes = 5
        self.thresholds = self._validate_thresholds(thresholds)
        self.model = nn.ModuleDict({
            name: RiverUNetPlusPlus(num_classes=2, encoder_weights=encoder_weights)
            for name in EXPERT_SPECS
        }).cpu()
        self.optimizer_states = {name: None for name in EXPERT_SPECS}
        self.class_weights = {name: None for name in EXPERT_SPECS}
        self.fitted_experts = set()
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False

    @staticmethod
    def _validate_thresholds(thresholds):
        if thresholds is None:
            return {name: 0.5 for name in EXPERT_SPECS}
        if set(thresholds) != set(EXPERT_SPECS):
            raise ValueError(f"Threshold keys must be exactly {tuple(EXPERT_SPECS)}")
        result = {name: float(thresholds[name]) for name in EXPERT_SPECS}
        if any(not math.isfinite(value) or not 0.0 < value < 1.0
               for value in result.values()):
            raise ValueError("Every threshold must be finite and between 0 and 1")
        return result

    @staticmethod
    def _check_expert(name):
        if name not in EXPERT_SPECS:
            raise ValueError(f"Unknown expert {name!r}; choose {tuple(EXPERT_SPECS)}")

    @staticmethod
    def to_binary_targets(name, masks):
        UNetPlusPlusEffB3RiverEnsemble._check_expert(name)
        positive = torch.zeros_like(masks, dtype=torch.bool)
        for class_id in EXPERT_SPECS[name]:
            positive |= masks == class_id
        return positive.long()

    def _compute_class_weights(self, name, train_loader):
        counts = torch.zeros(2, dtype=torch.float64)
        for _, masks in train_loader:
            binary = self.to_binary_targets(name, masks).reshape(-1).cpu()
            counts += torch.bincount(binary, minlength=2).double()
        if torch.any(counts == 0):
            raise ValueError(
                f"{name}: train fold must contain both rest and positive pixels; "
                f"got {counts.long().tolist()}"
            )
        frequencies = counts / counts.sum()
        weights = 1.0 / torch.sqrt(frequencies.clamp_min(1e-8))
        weights = (weights / weights.mean()).float()
        print(f"{name}: positive source IDs {EXPERT_SPECS[name]}")
        print("Binary pixel counts [rest, positive]:", counts.long().tolist())
        print("Binary weights [rest, positive]:", weights.tolist())
        return weights

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        for index, name in enumerate(EXPERT_SPECS, start=1):
            print(f"Expert {index}/{len(EXPERT_SPECS)}: {name}")
            self.fit_expert(name, train_loader, val_loader, save_best, patience)

    def fit_expert(self, name, train_loader, val_loader=None, save_best=True, patience=None):
        """Train one expert independently; permits separate training and bundling."""
        self._check_expert(name)
        if len(train_loader) == 0 or (val_loader is not None and len(val_loader) == 0):
            raise ValueError("Training and optional validation loaders must be nonempty")
        weights = self._compute_class_weights(name, train_loader)
        expert = self.model[name]
        self.fitted_experts.discard(name)
        self.is_fitted = False
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        start = time.time()

        try:
            expert.to(settings.DEVICE)
            # Construct after moving parameters; only this optimizer is on GPU.
            optimizer = optim.Adam(expert.parameters(), lr=self.LEARNING_RATE)
            if self.optimizer_states[name] is not None:
                optimizer.load_state_dict(self.optimizer_states[name])
            criterion = WeightedCEDiceLoss(weights.to(settings.DEVICE)).to(settings.DEVICE)

            for epoch in range(EPOCHS_COUNT):
                expert.train()
                train_loss = 0.0
                for images, masks in train_loader:
                    images = images.to(settings.DEVICE)
                    targets = self.to_binary_targets(name, masks).to(settings.DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(expert(images), targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                val_loss = None
                if val_loader is not None:
                    expert.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for images, masks in val_loader:
                            images = images.to(settings.DEVICE)
                            targets = self.to_binary_targets(name, masks).to(settings.DEVICE)
                            val_loss += criterion(expert(images), targets).item()
                    val_loss /= len(val_loader)

                if val_loss is None:
                    remaining = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
                    print(
                        f"{name} | Epoch {epoch + 1}/{EPOCHS_COUNT} "
                        f"| Train Loss: {train_loss:.4f} "
                        f"| Left approx {utils.beautify_time_left(remaining)}"
                    )
                else:
                    print(
                        f"{name} | Epoch {epoch + 1}/{EPOCHS_COUNT} "
                        f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                    )

                if save_best and val_loss is not None:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = (
                            _snapshot_to_cpu(expert.state_dict()),
                            _snapshot_to_cpu(optimizer.state_dict()),
                        )
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience is not None and patience_counter >= patience:
                            print(f"{name}: early stopping at epoch {epoch + 1}")
                            break

            if best_state is not None:
                expert.load_state_dict(best_state[0])
                optimizer.load_state_dict(best_state[1])
            optimizer.zero_grad(set_to_none=True)
            self.optimizer_states[name] = _snapshot_to_cpu(optimizer.state_dict())
            self.class_weights[name] = weights
            self.fitted_experts.add(name)
            self.is_fitted = self.fitted_experts == set(EXPERT_SPECS)
        finally:
            expert.zero_grad(set_to_none=True)
            expert.cpu()

    def predict_expert_probabilities(self, images):
        """Return four CPU [B,H,W] positive maps, not a five-class posterior."""
        probabilities = {}
        with torch.no_grad():
            images = images.to(settings.DEVICE)
            for name, expert in self.model.items():
                try:
                    expert.to(settings.DEVICE)
                    expert.eval()
                    probabilities[name] = torch.softmax(expert(images), dim=1)[:, 1].cpu()
                finally:
                    expert.cpu()
        return probabilities

    def fuse_probabilities(self, probabilities):
        """Our explicit fusion policy; it is not a method from the river paper."""
        if set(probabilities) != set(EXPERT_SPECS):
            raise ValueError("Fusion requires all four expert maps")
        shape = probabilities["arteries_crossings"].shape
        if len(shape) != 3 or any(
            value.shape != shape or value.device.type != "cpu"
            for value in probabilities.values()
        ):
            raise ValueError("Expert maps must have matching CPU [B,H,W] shapes")

        artery = probabilities["arteries_crossings"]
        vein = probabilities["veins_crossings"]
        crossing = probabilities["crossings"]
        class4 = probabilities["class4"]
        artery_positive = artery > self.thresholds["arteries_crossings"]
        vein_positive = vein > self.thresholds["veins_crossings"]
        class4_positive = class4 > self.thresholds["class4"]
        accepted_crossing = (
            (crossing > self.thresholds["crossings"])
            & artery_positive & vein_positive
        )

        # Competing non-crossing candidates; exact ties prefer ID 1, then 2, then 4.
        scores = torch.stack([
            artery.masked_fill(~artery_positive, -1.0),
            vein.masked_fill(~vein_positive, -1.0),
            class4.masked_fill(~class4_positive, -1.0),
        ], dim=1)
        winning_score, winner = scores.max(dim=1)
        labels = torch.tensor([1, 2, 4], dtype=torch.long)[winner]
        labels[winning_score < 0] = 0
        # A/V overlap alone never produces class 3. The direct crossing expert is required.
        labels[accepted_crossing] = 3
        return labels

    def predict(self, images):
        missing = set(EXPERT_SPECS) - self.fitted_experts
        if missing:
            raise RuntimeError(f"Train or load all four experts before predict(); missing {sorted(missing)}")
        return self.fuse_probabilities(self.predict_expert_probabilities(images))

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": ARCHITECTURE,
            "expert_specs": {name: list(ids) for name, ids in EXPERT_SPECS.items()},
            "num_classes": self.num_classes,
            "fusion_policy": FUSION_POLICY,
            "thresholds": dict(self.thresholds),
            "model_state_dict": self.model.state_dict(),
            "optimizer_states": self.optimizer_states,
            "class_weights": self.class_weights,
            "fitted_experts": sorted(self.fitted_experts),
            "LEARNING_RATE": self.LEARNING_RATE,
            "is_fitted": self.is_fitted,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_specs = {name: list(ids) for name, ids in EXPERT_SPECS.items()}
        if (
            checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("expert_specs") != expected_specs
            or checkpoint.get("num_classes") != self.num_classes
            or checkpoint.get("fusion_policy") != FUSION_POLICY
        ):
            raise ValueError("Expected a unified four-expert RIVER checkpoint")
        thresholds = self._validate_thresholds(checkpoint["thresholds"])
        if set(checkpoint["optimizer_states"]) != set(EXPERT_SPECS):
            raise ValueError("Checkpoint must contain optimizer entries for all four experts")
        if set(checkpoint["class_weights"]) != set(EXPERT_SPECS):
            raise ValueError("Checkpoint must contain loss entries for all four experts")
        fitted = set(checkpoint["fitted_experts"])
        if not fitted <= set(EXPERT_SPECS):
            raise ValueError("Checkpoint contains unknown fitted experts")
        self.model.cpu()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_states = checkpoint["optimizer_states"]
        self.class_weights = checkpoint["class_weights"]
        self.thresholds = thresholds
        self.fitted_experts = fitted
        self.LEARNING_RATE = checkpoint.get("LEARNING_RATE", LEARNING_RATE)
        self.is_fitted = fitted == set(EXPERT_SPECS)

    def save_expert(self, name, path):
        """Export an independently trained river expert for later assembly."""
        self._check_expert(name)
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": EXPERT_ARCHITECTURE,
            "expert_name": name,
            "positive_class_ids": list(EXPERT_SPECS[name]),
            "num_classes": 2,
            "threshold": self.thresholds[name],
            "model_state_dict": self.model[name].state_dict(),
            "optimizer_state_dict": self.optimizer_states[name],
            "class_weights": self.class_weights[name],
            "is_fitted": name in self.fitted_experts,
            "LEARNING_RATE": self.LEARNING_RATE,
        }, path)

    def load_expert(self, name, path):
        self._check_expert(name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("architecture") != EXPERT_ARCHITECTURE
            or checkpoint.get("expert_name") != name
            or checkpoint.get("positive_class_ids") != list(EXPERT_SPECS[name])
            or checkpoint.get("num_classes") != 2
            or checkpoint.get("LEARNING_RATE") != self.LEARNING_RATE
        ):
            raise ValueError(f"Expected a matching independently trained RIVER expert: {name}")
        proposed = dict(self.thresholds)
        proposed[name] = checkpoint["threshold"]
        proposed = self._validate_thresholds(proposed)
        self.model[name].cpu()
        self.model[name].load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_states[name] = checkpoint["optimizer_state_dict"]
        self.class_weights[name] = checkpoint["class_weights"]
        self.thresholds = proposed
        self.fitted_experts.discard(name)
        if checkpoint.get("is_fitted", False):
            self.fitted_experts.add(name)
        self.is_fitted = self.fitted_experts == set(EXPERT_SPECS)

    @staticmethod
    def get_model_save_path(fold_num):
        return MODEL_DIR / f"weights_{fold_num}.eth"

    @staticmethod
    def get_interval_metrics_save_path():
        return MODEL_DIR / "interval_metrics.csv"

    @staticmethod
    def get_raw_metrics_save_path():
        return MODEL_DIR / "raw_metrics.csv"

    @staticmethod
    def get_model_name():
        return "UN-WCD-RIVER-ENSEMBLE"

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image * np.asarray(STD) + np.asarray(MEAN), 0, 1)
        truth = self._mask_to_rgb(mask_tensor)
        prediction = self._mask_to_rgb(self.predict(image_tensor.unsqueeze(0))[0])
        for axis, data in ((ax_image, image), (ax_truth, truth), (ax_pred, prediction)):
            axis.imshow(data)
            axis.axis("off")

    @staticmethod
    def _mask_to_rgb(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in settings.COLOR_MAP.items():
            rgb[mask == class_id] = color
        return rgb
