# eval_external_patches.py
import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import settings
from common import metrics


# --- ВЫБОР МОДЕЛИ ---
from UNETPP_DF.model import ImprovedUNetPlusPlus, val_extra_transforms
ModelClass = ImprovedUNetPlusPlus


# --- ПУТИ ---
EXTERNAL_ROOT = settings.BASE_DIR / "external_datasets"
OUTPUT_ROOT = settings.BASE_DIR / "external_eval"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

MODEL_NAME = ModelClass.get_model_name()
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / MODEL_NAME
MODEL_OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)


# --- ПАРАМЕТРЫ НОРМАЛИЗАЦИИ (как в datasets_exp.py) ---
FOV_THRESHOLD = 10
MARGIN = 5
PAD_MODE = "constant"
TARGET_SIZE = (1280, 1280)   # (W, H)
FOV_DETECT_SCALE = 0.25

# --- ПАРАМЕТРЫ НАРЕЗКИ (как в dataset_cut.py, без overlap) ---
PATCH_SIZE = 256
STRIDE = 256

# --- ИНФЕРЕНС ---
INFER_BATCH_SIZE = 8

# --- НОРМАЛИЗАЦИЯ ДЛЯ МОДЕЛИ (та же, что при обучении) ---
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# =====================================================================
#                        НОРМАЛИЗАЦИЯ (datasets_exp.py)
# =====================================================================

def find_fov_bbox_fast(img_bgr, threshold, margin, scale):
    H, W = img_bgr.shape[:2]
    if scale < 1.0:
        small = cv2.resize(
            img_bgr,
            (max(1, int(W * scale)), max(1, int(H * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = img_bgr
        scale = 1.0

    gray = small.max(axis=2)
    mask = (gray > threshold).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0, 0, W, H

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))

    inv = 1.0 / scale
    x = int(stats[best, cv2.CC_STAT_LEFT] * inv)
    y = int(stats[best, cv2.CC_STAT_TOP] * inv)
    w = int(stats[best, cv2.CC_STAT_WIDTH] * inv)
    h = int(stats[best, cv2.CC_STAT_HEIGHT] * inv)

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(W - x, w + 2 * margin)
    h = min(H - y, h + 2 * margin)
    return x, y, w, h


def crop_and_square(img, bbox, pad_mode):
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)

    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left

    border = cv2.BORDER_REFLECT_101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT

    if crop.ndim == 2:
        return cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, border, value=0
        )
    return cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, border, value=(0, 0, 0)
    )


def resize_multiclass_mask(mask, size, num_classes):
    """Индексная маска -> ресайз через по-канальный one-hot -> argmax."""
    out_w, out_h = size
    acc = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    for c in range(num_classes):
        ch = (mask == c).astype(np.uint8) * 255
        acc[c] = cv2.resize(ch, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.argmax(acc, axis=0).astype(mask.dtype)


def rgb_to_labels_rgb(mask_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    """RGB-маска (H,W,3) -> индексная (H,W). color_map: {class_idx: [r,g,b]}."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for class_idx, color in color_map.items():
        m = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        out[m] = class_idx
    return out


def normalize_image_and_mask(image_rgb: np.ndarray, mask_rgb: np.ndarray):
    """
    image_rgb : HxWx3 uint8 RGB — оригинальная картинка
    mask_rgb  : HxWx3 uint8 RGB — оригинальная цветная маска
    Возвращает:
        img_out       : 1280x1280x3 uint8 RGB — нормализованная картинка
        mask_idx_out  : 1280x1280   uint8      — индексная маска
    """
    # cv2 работает в BGR, поэтому конвертим туда-обратно.
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    # Цветная маска BGR -> индексная (до ресайза)
    # Внимание: settings.COLOR_MAP задан в RGB, а mask_bgr — в BGR.
    # Конвертируем COLOR_MAP в BGR для корректного сравнения.
    color_map_bgr = {
        cls: [color[2], color[1], color[0]]
        for cls, color in settings.COLOR_MAP.items()
    }
    mask_idx = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for cls, color in color_map_bgr.items():
        m = np.all(mask_bgr == np.array(color, dtype=np.uint8), axis=-1)
        mask_idx[m] = cls

    # FOV bbox по картинке
    bbox = find_fov_bbox_fast(img_bgr, FOV_THRESHOLD, MARGIN, FOV_DETECT_SCALE)

    # Crop + square pad
    img_crop = crop_and_square(img_bgr, bbox, PAD_MODE)
    mask_crop = crop_and_square(mask_idx, bbox, PAD_MODE)

    # Resize
    if img_crop.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        img_out_bgr = cv2.resize(img_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    else:
        img_out_bgr = img_crop

    if mask_crop.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        mask_idx_out = resize_multiclass_mask(mask_crop, TARGET_SIZE, settings.NUM_CLASSES)
    else:
        mask_idx_out = mask_crop

    img_out_rgb = cv2.cvtColor(img_out_bgr, cv2.COLOR_BGR2RGB)
    return img_out_rgb, mask_idx_out


# =====================================================================
#                             DATASET
# =====================================================================

class ExternalPatchDataset(Dataset):
    """
    Один элемент = один патч 256x256.
    Атрибуты:
        patch_tensors : list[torch.Tensor] CHW float  — нормализованные патчи
        meta          : list[dict]                     — координаты и путь
    """
    def __init__(self, img_1280: np.ndarray, mask_idx_1280: np.ndarray,
                 rel_path: str, patch_size: int = 256, stride: int = 256):
        self.img = img_1280
        self.mask = mask_idx_1280
        self.rel_path = rel_path
        self.patch_size = patch_size
        self.stride = stride

        H, W = self.img.shape[:2]
        coords = []
        y = 0
        while y + patch_size <= H:
            x = 0
            while x + patch_size <= W:
                coords.append((y, x))
                x += stride
            y += stride
        # на случай, если H/W не кратны patch_size — добавляем крайние
        if coords:
            last_y, last_x = coords[-1]
            if last_y + patch_size < H:
                for x in range(0, W - patch_size + 1, stride):
                    coords.append((H - patch_size, x))
            if last_x + patch_size < W:
                for y in range(0, H - patch_size + 1, stride):
                    coords.append((y, W - patch_size))
        self.coords = coords

        # Нормализация Albumentations (только Normalize + ToTensorV2)
        self.transform = A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        ps = self.patch_size

        img_patch = self.img[y:y + ps, x:x + ps]        # HxWx3 uint8 RGB
        mask_patch = self.mask[y:y + ps, x:x + ps]      # HxW uint8

        aug = self.transform(image=img_patch, mask=mask_patch)
        img_t = aug["image"]          # CHW float
        mask_t = aug["mask"].long()   # HW long
        return img_t, mask_t, y, x


# =====================================================================
#                             ПАЙПЛАЙН
# =====================================================================

def infer_patches_for_image(model, img_1280: np.ndarray, mask_idx_1280: np.ndarray,
                            rel_path: str, infer_batch_size: int):
    """
    Прогоняет все патчи одной картинки через модель.
    Возвращает:
        pred_full : 1280x1280 uint8  — склеенная маска предсказаний
        gt_full   : 1280x1280 uint8  — GT (тот же, что на входе, но для симметрии)
    """
    ds = ExternalPatchDataset(img_1280, mask_idx_1280, rel_path,
                              patch_size=PATCH_SIZE, stride=STRIDE)
    loader = DataLoader(ds, batch_size=infer_batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    H, W = img_1280.shape[:2]
    pred_full = np.zeros((H, W), dtype=np.uint8)

    for imgs, masks, ys, xs in loader:
        preds = model.predict(imgs)  # BxHxW long cpu
        for b in range(imgs.size(0)):
            y = int(ys[b])
            x = int(xs[b])
            pred_full[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = (
                preds[b].cpu().numpy().astype(np.uint8)
            )

    return pred_full, mask_idx_1280


def evaluate_dataset(model, dataset_dir: Path, num_classes, class_names,
                     vis_dir_base: Path, infer_batch_size: int):
    """
    Прогоняет одну модель по всему датасету.
    Возвращает (per_image_rows, agg_metrics).
    """
    vis_dir_base.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    dataset_name = dataset_dir.name

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def collect(d):
        paths = []
        for e in exts:
            paths.extend(d.glob(f"*{e}"))
        return sorted(paths)

    img_paths = collect(images_dir)
    mask_paths = collect(masks_dir)
    if not img_paths or not mask_paths:
        raise FileNotFoundError(f"Пусто в {images_dir} или {masks_dir}")

    masks_by_stem = {p.stem: p for p in mask_paths}

    per_image_rows = []
    all_preds = []
    all_targets = []

    for img_path in img_paths:
        mask_path = masks_by_stem.get(img_path.stem)
        if mask_path is None:
            print(f"  [skip] нет маски для {img_path.name}")
            continue

        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        img_1280, mask_idx_1280 = normalize_image_and_mask(image_rgb, mask_rgb)

        pred_full, gt_full = infer_patches_for_image(
            model, img_1280, mask_idx_1280,
            rel_path=f"{dataset_name}/{img_path.name}",
            infer_batch_size=infer_batch_size,
        )

        # --- метрики по одной картинке (на полном разрешении 1280x1280) ---
        pred_flat = pred_full.reshape(-1)
        target_flat = gt_full.reshape(-1)
        m = metrics.get_all_metrics(
            pred_flat, target_flat,
            num_classes=num_classes,
            class_names=class_names,
        )
        m["image_path"] = f"{dataset_name}/{img_path.name}"
        per_image_rows.append(m)

        # --- визуализации ---
        stem = img_path.stem
        Image.fromarray(mask_to_rgb(gt_full)).save(vis_dir_base / f"{stem}_gt.png")
        Image.fromarray(mask_to_rgb(pred_full)).save(vis_dir_base / f"{stem}_pred.png")
        Image.fromarray(make_overlay(img_1280, pred_full)).save(
            vis_dir_base / f"{stem}_overlay.png"
        )

        all_preds.append(pred_flat)
        all_targets.append(target_flat)

    if not all_preds:
        raise RuntimeError(f"Не обработано ни одной картинки в {dataset_dir}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    agg_metrics = metrics.get_all_metrics(
        all_preds, all_targets,
        num_classes=num_classes,
        class_names=class_names,
    )
    return per_image_rows, agg_metrics


# =====================================================================
#                        ВИЗУАЛИЗАЦИЯ
# =====================================================================

def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in settings.COLOR_MAP.items():
        rgb[mask_np == cls] = np.array(color, dtype=np.uint8)
    return rgb


def make_overlay(image_rgb: np.ndarray, mask_np: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    color_mask = mask_to_rgb(mask_np).astype(np.float32)
    non_bg = (mask_np != 0)[..., None]
    blended = overlay * (1 - alpha * non_bg) + color_mask * (alpha * non_bg)
    return np.clip(blended, 0, 255).astype(np.uint8)


# =====================================================================
#                             MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--infer-batch-size", type=int, default=INFER_BATCH_SIZE)
    args = parser.parse_args()

    if args.datasets:
        dataset_dirs = [EXTERNAL_ROOT / name for name in args.datasets]
    else:
        dataset_dirs = [p for p in sorted(EXTERNAL_ROOT.iterdir()) if p.is_dir()]

    if not dataset_dirs:
        raise FileNotFoundError(f"В {EXTERNAL_ROOT} нет подпапок с датасетами")

    print(f"Модель: {MODEL_NAME}")
    print(f"Результаты: {MODEL_OUTPUT_ROOT}")
    print("Датасеты:")
    for d in dataset_dirs:
        print("  -", d.name)

    # --- модели всех фолдов ---
    models = []
    for fold in range(1, settings.FOLDS_NUM + 1):
        model = ModelClass()
        path = ModelClass.get_model_save_path(fold)
        print(f"Загрузка модели фолда {fold} из {path}")
        model.load(str(path))
        models.append(model)

    # --- по каждому датасету ---
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        out_dir = MODEL_OUTPUT_ROOT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## Датасет: {dataset_name} ##########")
        print(f"Результаты: {out_dir}")

        all_aggregates = []

        for fold_idx, model in enumerate(models, start=1):
            print(f"\n=== Фолд {fold_idx} ===")

            vis_dir = out_dir / "visualizations" / f"fold_{fold_idx}"
            if args.no_vis:
                vis_dir = None

            per_image_rows, agg_metrics = evaluate_dataset(
                model, dataset_dir,
                num_classes=settings.NUM_CLASSES,
                class_names=settings.CLASS_NAMES,
                vis_dir_base=vis_dir,
                infer_batch_size=args.infer_batch_size,
            )

            per_image_df = pd.DataFrame(per_image_rows)
            meta_cols = ["image_path"]
            other = [c for c in per_image_df.columns if c not in meta_cols]
            per_image_df = per_image_df[meta_cols + other]
            per_image_path = out_dir / f"per_image_fold_{fold_idx}.csv"
            per_image_df.to_csv(per_image_path, index=False)

            agg_metrics["fold"] = fold_idx
            all_aggregates.append(agg_metrics)

            print(f"  per-image метрики: {per_image_path}")
            if vis_dir is not None:
                print(f"  визуализации:      {vis_dir}")

        agg_df = pd.DataFrame(all_aggregates)
        meta_cols = ["fold"]
        other = [c for c in agg_df.columns if c not in meta_cols]
        agg_df = agg_df[meta_cols + other]
        agg_path = out_dir / "aggregate_all_folds.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"\nСводный агрегат по датасету: {agg_path}")

    # --- сводка по всем датасетам ---
    all_dataset_aggs = []
    for dataset_dir in dataset_dirs:
        agg_path = MODEL_OUTPUT_ROOT / dataset_dir.name / "aggregate_all_folds.csv"
        if agg_path.exists():
            df = pd.read_csv(agg_path)
            df.insert(0, "dataset", dataset_dir.name)
            all_dataset_aggs.append(df)

    if all_dataset_aggs:
        combined = pd.concat(all_dataset_aggs, ignore_index=True)
        meta_cols = ["dataset", "fold"]
        other = [c for c in combined.columns if c not in meta_cols]
        combined = combined[meta_cols + other]
        combined_path = MODEL_OUTPUT_ROOT / "aggregate_all_datasets.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nСводка по всем датасетам: {combined_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
