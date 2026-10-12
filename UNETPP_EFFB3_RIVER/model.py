"""Standalone UNETPP_EFFB3 adaptation of river segmentation attention fusion."""

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
from UNETPP_EFFB3_RIVER.river_attention import RiverUNetPlusPlus


IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_DIR = pathlib.Path("UNETPP_EFFB3_RIVER")
ARCHITECTURE = "unetplusplus-efficientnet-b3-river-raff-cbam-v1"

# Identical to UNETPP_EFFB3; common.data_preparation adds ToTensorV2.
train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]
val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class WeightedCEDiceLoss(nn.Module):
    """Baseline weighted multiclass CE + Dice, both applied to logits."""

    def __init__(self, class_weights=None, ce_coef=1.0, dice_coef=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_coef = ce_coef
        self.dice_coef = dice_coef

    def forward(self, logits, targets):
        return self.ce_coef * self.ce(logits, targets) + self.dice_coef * self.dice(logits, targets)


def _snapshot_to_cpu(value):
    """Clone model/optimizer state without keeping the best checkpoint on GPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return value


class UNetPlusPlusEffB3River(BaseModel):
    """Same pipeline contract as UNETPP_EFFB3, with independently owned logic.

    encoder_weights=None allows offline construction before loading a checkpoint.
    The default training initialization remains ImageNet, as in the baseline.
    """

    def __init__(self, encoder_weights="imagenet"):
        super().__init__()
        self.num_classes = settings.NUM_CLASSES
        self.model = RiverUNetPlusPlus(
            num_classes=self.num_classes, encoder_weights=encoder_weights
        ).to(settings.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = WeightedCEDiceLoss().to(settings.DEVICE)
        self.LEARNING_RATE = LEARNING_RATE
        self.class_weights = None
        self.is_fitted = False

    def _compute_class_weights(self, train_loader):
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        for _, masks in train_loader:
            flat = masks.reshape(-1).cpu()
            counts += torch.bincount(flat, minlength=self.num_classes).double()

        freq = counts / counts.sum().clamp_min(1.0)
        weights = 1.0 / torch.sqrt(freq.clamp_min(1e-8))
        weights = (weights / weights.mean()).float().to(settings.DEVICE)
        print("Class pixel counts:", counts.long().tolist())
        print("Class weights:", [round(x, 4) for x in weights.detach().cpu().tolist()])
        return weights

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        if len(train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch")
        if val_loader is not None and len(val_loader) == 0:
            raise ValueError("val_loader must contain at least one batch")

        self.class_weights = self._compute_class_weights(train_loader)
        self.criterion = WeightedCEDiceLoss(self.class_weights).to(settings.DEVICE)
        best_val_loss = float("inf")
        best_state = None
        best_optimizer_state = None
        patience_counter = 0
        start = time.time()

        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(settings.DEVICE)
                masks = masks.to(settings.DEVICE)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
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
                        masks = masks.to(settings.DEVICE)
                        loss = self.criterion(self.model(images), masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} "
                    f"| Val Loss: {val_loss:.4f}"
                )
            else:
                approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
                time_left_str = utils.beautify_time_left(approx_time_left)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} "
                    f"| Left approx {time_left_str}"
                )

            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _snapshot_to_cpu(self.model.state_dict())
                best_optimizer_state = _snapshot_to_cpu(self.optimizer.state_dict())
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and val_loss is not None and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.optimizer.load_state_dict(best_optimizer_state)
        self.is_fitted = True

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(settings.DEVICE))
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu()

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": ARCHITECTURE,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_classes": self.num_classes,
                "LEARNING_RATE": self.LEARNING_RATE,
                "class_weights": _snapshot_to_cpu(self.class_weights),
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path):
        # load_state_dict moves model weights and Adam moments to their devices;
        # non-capturable Adam step counters should remain on CPU.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Checkpoint is not a UNETPP_EFFB3_RIVER model")
        if checkpoint.get("num_classes") != self.num_classes:
            raise ValueError("Checkpoint class count does not match settings.NUM_CLASSES")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.LEARNING_RATE = checkpoint.get("LEARNING_RATE", self.LEARNING_RATE)
        self.class_weights = checkpoint.get("class_weights")
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(settings.DEVICE)
        self.criterion = WeightedCEDiceLoss(self.class_weights).to(settings.DEVICE)
        self.is_fitted = checkpoint.get("is_fitted", True)

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return MODEL_DIR / f"weights_{fold_num}.eth"

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return MODEL_DIR / "interval_metrics.csv"

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return MODEL_DIR / "raw_metrics.csv"

    @staticmethod
    def get_model_name():
        return "UN-WCD-RIVER"

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        img = self._denormalize(image_tensor)
        true_rgb = self._mask_to_rgb(mask_tensor)
        pred = self.predict(image_tensor.unsqueeze(0)).squeeze(0)
        pred_rgb = self._mask_to_rgb(pred)
        ax_image.imshow(img)
        ax_truth.imshow(true_rgb)
        ax_pred.imshow(pred_rgb)
        for axis in (ax_image, ax_truth, ax_pred):
            axis.axis("off")

    def _denormalize(self, img_tensor):
        img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)

    def _mask_to_rgb(self, mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb
