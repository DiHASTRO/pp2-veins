# viz_one_patch.py
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import common.settings_patches as settings
from UNETPP_WCD.model import ImprovedUNetPlusPlus


# ---------- ПАРАМЕТРЫ ----------
PATCH_STEM = "001_G_y256_x512"   # <-- подставь свой
FOLD = 1
MEAN = (0.485, 0.456, 0.406)   # как в train_extra_transforms
STD  = (0.229, 0.224, 0.225)
# --------------------------------


PATCH_IMG = settings.DATASET_IMG_DIR  / f"{PATCH_STEM}.png"
PATCH_MSK = settings.DATASET_MASK_DIR / f"{PATCH_STEM}.png"


# ---------- Утилиты ----------
def rgb_to_labels(mask_rgb, color_map):
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in color_map.items():
        out[np.all(mask_rgb == np.array(color, dtype=mask_rgb.dtype), axis=-1)] = cls
    return out


def idx_to_rgb(idx, color_map):
    out = np.zeros((*idx.shape, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        out[idx == cls] = color
    return out


def to_green_canvas(img_rgb):
    """
    RGB (H,W,3) uint8 -> 'зелёное' изображение (H,W,3) uint8.
    Оставляем только зелёный канал, красный и синий = 0.
    """
    gray = img_rgb.mean(axis=2).astype(np.uint8)   # яркость
    out = np.zeros_like(img_rgb)
    out[..., 1] = gray                              # G-канал
    return out


def normalize_for_model(img_rgb, mean, std):
    """(H,W,3) uint8 -> (1,3,H,W) float32, нормализовано как в обучении."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return t


# ---------- Загрузка ----------
img = np.array(Image.open(PATCH_IMG).convert("RGB"))
msk_rgb = np.array(Image.open(PATCH_MSK).convert("RGB"))
gt_idx = rgb_to_labels(msk_rgb, settings.COLOR_MAP)

# Зелёное изображение для визуализации
img_green = to_green_canvas(img)

# Тензор для модели — нормализованный RGB (в зелёный канал НЕ переводим!)
# ВАЖНО: модель обучалась на полном RGB. Для визуализации зеленеем,
# для инференса — нет.
tensor = normalize_for_model(img, MEAN, STD)

# ---------- Модель ----------
model = ImprovedUNetPlusPlus()
model.load(ImprovedUNetPlusPlus.get_model_save_path(FOLD))

preds = model.predict(tensor)            # (1,H,W) CPU
if preds.dim() == 4:
    preds = preds.argmax(dim=1)
pred_idx = preds.squeeze(0).cpu().numpy().astype(np.uint8)

pred_rgb = idx_to_rgb(pred_idx, settings.COLOR_MAP)
gt_rgb_show = idx_to_rgb(gt_idx, settings.COLOR_MAP)

# ---------- Визуализация ----------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_green);     axes[0].set_title("image (green channel)")
axes[1].imshow(gt_rgb_show);   axes[1].set_title("GT")
axes[2].imshow(pred_rgb);      axes[2].set_title("pred")

# Наложение: зелёный фон + цветная маска предсказания полупрозрачно
axes[3].imshow(img_green)
axes[3].imshow(pred_rgb, alpha=0.5)
axes[3].set_title("image (green) + pred overlay")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("viz_one_patch.png", dpi=120, bbox_inches="tight")
plt.show()

# ---------- Dice ----------
def dice_per_class(pred, gt, num_classes):
    res = {}
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        res[settings.CLASS_NAMES.get(c, f"class_{c}")] = (
            2 * inter / denom if denom > 0 else float("nan")
        )
    return res

print("Dice по классам:")
for k, v in dice_per_class(pred_idx, gt_idx, settings.NUM_CLASSES).items():
    print(f"  {k:12s}: {v:.4f}")