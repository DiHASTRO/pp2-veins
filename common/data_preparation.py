import numpy as np
from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import settings


class SegmentationDataset(Dataset):
    def __init__(self, image_paths: List[Path], mask_paths: List[Path],
                 transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def rgb_to_labels(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-маску (H,W,3) в маску индексов классов (H,W) uint8.
        """
        h, w, _ = mask_rgb.shape
        mask_labels = np.zeros((h, w), dtype=np.uint8)
        for class_idx, color in settings.COLOR_MAP.items():
            # Сравниваем по всем каналам
            color_mask = np.all(mask_rgb == color, axis=-1)
            mask_labels[color_mask] = class_idx
        return mask_labels

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        # Загружаем маску как RGB
        mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
        # Преобразуем цветную маску в индексы классов
        mask = self.rgb_to_labels(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.long()
        return image, mask



def get_train_loader(train_extra: Optional[List[A.BasicTransform]] = None,
                     batch_size: int = None,
                     num_workers: int = None) -> DataLoader:
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = num_workers or settings.NUM_WORKERS

    img_paths = sorted(settings.TRAIN_IMG_DIR.glob("*.png"))
    mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
    assert len(img_paths) == len(mask_paths), "Число изображений и масок не совпадает"

    transforms = []
    if train_extra:
        transforms.extend(train_extra)
    transforms.append(ToTensorV2())

    transform = A.Compose(transforms)
    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    return loader


def get_test_loader(extra_transforms: Optional[List[A.BasicTransform]] = None,
                    batch_size: int = None,
                    num_workers: int = None) -> DataLoader:
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = num_workers or settings.NUM_WORKERS

    img_paths = sorted(settings.TEST_IMG_DIR.glob("*.png"))
    mask_paths = sorted(settings.TEST_MASK_DIR.glob("*.png"))

    transforms = []
    if extra_transforms:
        transforms.extend(extra_transforms)
    transforms.append(ToTensorV2())

    transform = A.Compose(transforms)
    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader
