import pathlib
from collections import deque

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from common import settings
from common.base_model import BaseModel

NUM_CLASSES = settings.NUM_CLASSES
IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ARTERY_CLASS = 1
VEIN_CLASS = 2
CROSSING_CLASS = 3
CAPILLARY_CLASS = 4

train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class MultiClassDiceLoss(nn.Module):
    def __init__(self, include_background: bool = False, smooth: float = 1.0):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        denominator = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class ConsistentDeepLabV3Plus(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
        ).to(settings.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.dice_loss = MultiClassDiceLoss(include_background=False)
        self.class_weights = None
        self.learning_rate = LEARNING_RATE
        self.is_fitted = False

    def _rgb_to_labels(self, mask_rgb: np.ndarray) -> np.ndarray:
        mask_labels = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for class_idx, color in settings.COLOR_MAP.items():
            color_mask = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
            mask_labels[color_mask] = class_idx
        return mask_labels

    def _compute_class_weights(self, train_loader) -> torch.Tensor:
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        dataset = train_loader.dataset

        for mask_path in dataset.mask_paths:
            mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
            mask = self._rgb_to_labels(mask_rgb)
            bincount = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
            counts += bincount

        frequencies = counts / counts.sum()
        weights = 1.0 / np.sqrt(frequencies + 1e-8)
        weights = weights / weights.mean()
        weights[0] *= 0.35
        return torch.tensor(weights, dtype=torch.float32, device=settings.DEVICE)

    def _criterion(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(outputs, masks, weight=self.class_weights)
        dice = self.dice_loss(outputs, masks)
        return 0.7 * ce + 0.3 * dice

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        if self.class_weights is None:
            self.class_weights = self._compute_class_weights(train_loader)
            print(f"Class weights: {self.class_weights.detach().cpu().numpy().round(4).tolist()}")

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self._criterion(outputs, masks)
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
                        images = images.to(settings.DEVICE, non_blocking=True)
                        masks = masks.to(settings.DEVICE, non_blocking=True)
                        outputs = self.model(images)
                        loss = self._criterion(outputs, masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

            if val_loss is not None:
                print(f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f}")

            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and val_loss is not None and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if save_best and best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(settings.DEVICE)

        self.is_fitted = True

    def _smooth_av_probabilities(self, probs: torch.Tensor) -> torch.Tensor:
        av_probs = probs[:, [ARTERY_CLASS, VEIN_CLASS]]
        smoothed = F.avg_pool2d(av_probs, kernel_size=3, stride=1, padding=1)
        return smoothed

    def _connected_components_majority_vote(self, pred_map: np.ndarray, artery_score: np.ndarray, vein_score: np.ndarray) -> np.ndarray:
        refined = pred_map.copy()
        av_mask = np.isin(pred_map, [ARTERY_CLASS, VEIN_CLASS])
        visited = np.zeros_like(av_mask, dtype=bool)
        h, w = pred_map.shape
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for y in range(h):
            for x in range(w):
                if not av_mask[y, x] or visited[y, x]:
                    continue

                queue = deque([(y, x)])
                visited[y, x] = True
                coords = []

                while queue:
                    cy, cx = queue.popleft()
                    coords.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and av_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                ys, xs = zip(*coords)
                ys = np.array(ys)
                xs = np.array(xs)

                artery_total = float(artery_score[ys, xs].sum())
                vein_total = float(vein_score[ys, xs].sum())
                dominance = max(artery_total, vein_total) / (artery_total + vein_total + 1e-8)
                dominant_class = ARTERY_CLASS if artery_total >= vein_total else VEIN_CLASS

                if len(coords) <= 64 or dominance >= 0.58:
                    refined[ys, xs] = dominant_class

        return refined

    def _refine_prediction(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        av_smoothed = self._smooth_av_probabilities(probs)
        av_choice = torch.argmax(av_smoothed, dim=1) + ARTERY_CLASS
        av_mask = (pred == ARTERY_CLASS) | (pred == VEIN_CLASS)
        pred = torch.where(av_mask, av_choice, pred)

        refined_batch = []
        probs_np = probs.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        for i in range(pred_np.shape[0]):
            refined = self._connected_components_majority_vote(
                pred_np[i],
                probs_np[i, ARTERY_CLASS],
                probs_np[i, VEIN_CLASS],
            )
            refined_batch.append(torch.from_numpy(refined).long())

        return torch.stack(refined_batch, dim=0)

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE)
            outputs = self.model(images)
            preds = self._refine_prediction(outputs)
        return preds.cpu()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_weights': None if self.class_weights is None else self.class_weights.detach().cpu(),
            'num_classes': NUM_CLASSES,
            'learning_rate': self.learning_rate,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stored_weights = checkpoint.get('class_weights')
        if stored_weights is not None:
            self.class_weights = stored_weights.to(settings.DEVICE)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.is_fitted = True

    @staticmethod
    def get_model_save_path() -> pathlib.Path:
        return pathlib.Path("ADVANCED_SEGM_V3Plus/weights.pth")

    @staticmethod
    def get_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("ADVANCED_SEGM_V3Plus/metrics.csv")

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        img = self._denormalize(image_tensor)
        true_rgb = self._mask_to_rgb(mask_tensor)
        with torch.no_grad():
            pred = self.predict(image_tensor.unsqueeze(0)).squeeze(0).cpu()
        pred_rgb = self._mask_to_rgb(pred)

        ax_image.imshow(img)
        ax_image.axis('off')
        ax_truth.imshow(true_rgb)
        ax_truth.axis('off')
        ax_pred.imshow(pred_rgb)
        ax_pred.axis('off')

    def _denormalize(self, img_tensor):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def _mask_to_rgb(self, mask_tensor):
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb