import numpy as np
import typing as tp
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


def create_cross_val_loaders(
    train_extra_transforms: Optional[List[A.BasicTransform]] = None,
    val_extra_transforms: Optional[List[A.BasicTransform]] = None,
    batch_size: int = None,
    num_workers: int = None,
) -> List[tp.Tuple[DataLoader, DataLoader]]:
    """
    Создаёт список пар (train_loader, test_loader) для кросс-валидации.
    
    Параметры:
        train_extra: дополнительные аугментации для тренировочных данных
        extra_transforms: дополнительные аугментации для тестовых данных
        batch_size, num_workers: если None, берутся из settings
        num_folds: количество фолдов (по умолчанию 3)
        test_size: количество изображений в тестовой выборке на фолд (по умолчанию 20)
    
    Возвращает:
        Список кортежей длины num_folds: [(train_loader, test_loader), ...]
    """
    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE

    # Загрузка всех путей из общих директорий
    all_img_paths = sorted(settings.DATASET_IMG_DIR.glob("*.png"))
    all_mask_paths = sorted(settings.DATASET_MASK_DIR.glob("*.png"))

    all_img_paths, all_mask_paths = _shuffle_both_equally(all_img_paths, all_mask_paths)

    assert len(all_img_paths) == len(all_mask_paths), "Число изображений и масок не совпадает"
    total = len(all_img_paths)
    
    if total < num_folds * test_size:
        raise ValueError(f"Всего {total} изображений, требуется {num_folds * test_size} для {num_folds} фолдов по {test_size}")
    
    # Параметры загрузки
    bs = batch_size or settings.BATCH_SIZE
    nw = num_workers or settings.NUM_WORKERS
    
    # Вспомогательная функция для создания train loader (аналогична get_train_loader)
    def make_train_loader(img_paths, mask_paths):
        transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
        if train_extra_transforms:
            transforms.extend(train_extra_transforms)
        transforms.append(ToTensorV2())
        transform = A.Compose(transforms)
        dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
        return DataLoader(dataset, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True)
    
    # Вспомогательная функция для создания test loader (аналогична get_test_loader)
    def make_test_loader(img_paths, mask_paths):
        transforms = []
        if val_extra_transforms:
            transforms.extend(val_extra_transforms)
        transforms.append(ToTensorV2())
        transform = A.Compose(transforms)
        dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
        return DataLoader(dataset, batch_size=bs, shuffle=False,
                          num_workers=nw, pin_memory=True)
    
    result = []
    for fold in range(num_folds):
        start = fold * test_size
        end = start + test_size
        
        test_imgs = all_img_paths[start:end]
        test_masks = all_mask_paths[start:end]
        
        train_imgs = all_img_paths[:start] + all_img_paths[end:]
        train_masks = all_mask_paths[:start] + all_mask_paths[end:]
        
        train_loader = make_train_loader(train_imgs, train_masks)
        test_loader = make_test_loader(test_imgs, test_masks)
        result.append((train_loader, test_loader))
    
    return result


def _shuffle_both_equally(l1: tp.List, l2: tp.List) -> tp.Tuple[tp.List, tp.List]:
    assert len(l1) == len(l2)

    np.random.seed(settings.SEED)
    length = len(l1)
    indexes = np.random.randint(0, length, length)

    new_l1 = []
    new_l2 = []
    for i in indexes:
        new_l1.append(l1[i])
        new_l2.append(l2[i])

    return new_l1, new_l2
