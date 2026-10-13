import copy
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


IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]
val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class WeightedCEDiceLoss(nn.Module):
    """The baseline loss applied to two mutually exclusive output classes."""

    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits, binary_targets):
        return self.ce(logits, binary_targets) + self.dice(logits, binary_targets)


def _cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


class _UNetPlusPlusEffB3Binary(BaseModel):
    """RGB masks remain unchanged; each experiment maps its own positive IDs."""

    POSITIVE_CLASS_IDS = ()
    TARGET_CLASS_ID = None
    TARGET_NAME = None
    EXPERIMENT_NAME = None
    ARCHITECTURE = "UNETPP_EFFB3_BINARY"

    def __init__(self, encoder_weights="imagenet", threshold=0.5):
        super().__init__()
        if not self.POSITIVE_CLASS_IDS or self.TARGET_CLASS_ID is None:
            raise ValueError("Choose a concrete binary experiment class")
        self.threshold = self._validate_threshold(threshold)
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=2,
        ).to(settings.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = WeightedCEDiceLoss().to(settings.DEVICE)
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False

    @staticmethod
    def _validate_threshold(threshold):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be finite and strictly between 0 and 1")
        return threshold

    @classmethod
    def to_binary_targets(cls, masks):
        positive = torch.zeros_like(masks, dtype=torch.bool)
        for class_id in cls.POSITIVE_CLASS_IDS:
            positive |= masks == class_id
        return positive.long()

    @classmethod
    def targets_for_metrics(cls, masks):
        """Use the same union for evaluation as for training and prediction."""
        return cls.to_binary_targets(masks) * cls.TARGET_CLASS_ID

    def _compute_class_weights(self, train_loader):
        counts = torch.zeros(2, dtype=torch.float64)
        for _, masks in train_loader:
            binary = self.to_binary_targets(masks).reshape(-1).cpu()
            counts += torch.bincount(binary, minlength=2).double()
        if torch.any(counts == 0):
            raise ValueError(
                f"{self.EXPERIMENT_NAME}: train fold must contain both positive "
                f"and negative pixels; got {counts.long().tolist()}"
            )
        freq = counts / counts.sum()
        weights = 1.0 / torch.sqrt(freq.clamp_min(1e-8))
        weights = (weights / weights.mean()).float().to(settings.DEVICE)
        print(f"Positive source IDs: {self.POSITIVE_CLASS_IDS}")
        print("Binary pixel counts [rest, target]:", counts.long().tolist())
        print("Binary weights [rest, target]:", weights.cpu().tolist())
        return weights

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        if len(train_loader) == 0 or (val_loader is not None and len(val_loader) == 0):
            raise ValueError("Training and optional validation loaders must be nonempty")
        weights = self._compute_class_weights(train_loader)
        self.criterion = WeightedCEDiceLoss(weights).to(settings.DEVICE)
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        start = time.time()

        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(settings.DEVICE)
                targets = self.to_binary_targets(masks).to(settings.DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(settings.DEVICE)
                        targets = self.to_binary_targets(masks).to(settings.DEVICE)
                        val_loss += self.criterion(self.model(images), targets).item()
                val_loss /= len(val_loader)

            if val_loss is None:
                remaining = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} "
                    f"| Left approx {utils.beautify_time_left(remaining)}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} "
                    f"| Val Loss: {val_loss:.4f}"
                )

            if save_best and val_loss is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = (
                        _cpu_snapshot(self.model.state_dict()),
                        _cpu_snapshot(self.optimizer.state_dict()),
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state[0])
            self.optimizer.load_state_dict(best_state[1])
        self.is_fitted = True

    def predict_probabilities(self, images):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images.to(settings.DEVICE))
            return torch.softmax(logits, dim=1)[:, 1].cpu()

    def predict_binary(self, images):
        return (self.predict_probabilities(images) > self.threshold).long()

    def predict(self, images):
        # Keep the BaseModel label-ID contract, including the batch dimension at B=1.
        return self.predict_binary(images) * self.TARGET_CLASS_ID

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": self.ARCHITECTURE,
            "positive_class_ids": list(self.POSITIVE_CLASS_IDS),
            "target_class_id": self.TARGET_CLASS_ID,
            "num_classes": 2,
            "threshold": self.threshold,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_weights": _cpu_snapshot(self.criterion.ce.weight),
            "LEARNING_RATE": self.LEARNING_RATE,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("architecture") != self.ARCHITECTURE
            or checkpoint.get("num_classes") != 2
            or checkpoint.get("positive_class_ids") != list(self.POSITIVE_CLASS_IDS)
            or checkpoint.get("target_class_id") != self.TARGET_CLASS_ID
        ):
            raise ValueError(f"Checkpoint does not match {self.EXPERIMENT_NAME}")
        threshold = self._validate_threshold(checkpoint["threshold"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.criterion = WeightedCEDiceLoss(checkpoint.get("class_weights")).to(settings.DEVICE)
        self.threshold = threshold
        self.LEARNING_RATE = checkpoint.get("LEARNING_RATE", LEARNING_RATE)
        self.is_fitted = True

    @classmethod
    def _output_dir(cls):
        return pathlib.Path(__file__).resolve().parent / cls.EXPERIMENT_NAME

    @classmethod
    def get_model_save_path(cls, fold_num):
        return cls._output_dir() / f"weights_{fold_num}.eth"

    @classmethod
    def get_interval_metrics_save_path(cls):
        return cls._output_dir() / "interval_metrics.csv"

    @classmethod
    def get_raw_metrics_save_path(cls):
        return cls._output_dir() / "raw_metrics.csv"

    @classmethod
    def get_model_name(cls):
        return f"UN-WCD-BINARY-{cls.EXPERIMENT_NAME.upper()}"

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        img = image_tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * np.asarray(STD) + np.asarray(MEAN), 0, 1)
        truth = self.targets_for_metrics(mask_tensor).cpu().numpy()
        pred = self.predict(image_tensor.unsqueeze(0))[0].numpy()
        for ax, data in ((ax_image, img), (ax_truth, self._mask_to_rgb(truth)),
                         (ax_pred, self._mask_to_rgb(pred))):
            ax.imshow(data)
            ax.axis("off")

    @staticmethod
    def _mask_to_rgb(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in settings.COLOR_MAP.items():
            rgb[mask == class_id] = color
        return rgb


class UNetPlusPlusEffB3VeinsCrossings(_UNetPlusPlusEffB3Binary):
    POSITIVE_CLASS_IDS = (2, 3)
    TARGET_CLASS_ID = 2
    TARGET_NAME = "veins"
    EXPERIMENT_NAME = "veins_crossings"


class UNetPlusPlusEffB3ArteriesCrossings(_UNetPlusPlusEffB3Binary):
    POSITIVE_CLASS_IDS = (1, 3)
    TARGET_CLASS_ID = 1
    TARGET_NAME = "arteries"
    EXPERIMENT_NAME = "arteries_crossings"


class UNetPlusPlusEffB3Crossings(_UNetPlusPlusEffB3Binary):
    POSITIVE_CLASS_IDS = (3,)
    TARGET_CLASS_ID = 3
    TARGET_NAME = "crossings"
    EXPERIMENT_NAME = "crossings"
