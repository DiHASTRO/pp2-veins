import datetime as dt
import time
import typing as tp

import numpy as np
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A

from common import settings
from common import utils
from common.base_model import BaseModel

from UNETPP_ARTERIES import losses


# Константы (могут быть изменены)
NUM_CLASSES = 5
IMG_SIZE = 1024
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Константы для модели
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Дополнительные трансформации для этой модели (без ToTensorV2, его добавит data_preparation)
train_extra_transforms = [
    # A.ToFloat(max_value=255.0),
    A.Lambda(image=lambda img, **kwargs: img[:, :, 1:2]),

    # A.CLAHE(p=1.0),

    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),  # 1=LINEAR, 0=NEAREST
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    # A.ToFloat(max_value=255.0),
    A.Lambda(image=lambda img, **kwargs: img[:, :, 1:2]),

    # A.CLAHE(p=1.0),

    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),  # 1=LINEAR, 0=NEAREST
    A.Normalize(mean=MEAN, std=STD),
]


class UnetPlusPlus(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(in_channels=1).to(settings.DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = smp.losses.DiceLoss(mode='binary')
        self.LEARNING_RATE = LEARNING_RATE

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        avg_sec_per_epoch = 0
        for epoch_i in range(1, EPOCHS_COUNT + 1):
            epoch_start = time.time()

            # Обучение
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                masks: torch.Tensor
                from PIL import Image
                arr = masks[0].cpu().numpy().astype('uint8')
                Image.fromarray(np.dstack([(arr==1)*255, (arr==3)*255, np.zeros_like(arr)]).astype(np.uint8)).show()
                masks = self._get_mask_only_for([1, 3], masks)

                print(masks[0].shape)
                print(masks[0].cpu().numpy().shape)
                
                Image.fromarray(masks[0].cpu().numpy().astype('uint8') * 255, mode='L').show()

                exit()

                images, masks = images.to(settings.DEVICE), masks.to(settings.DEVICE)
                self.optimizer.zero_grad()

                masks = masks.unsqueeze(1)
                outputs = torch.sigmoid(self.model(images))
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Валидация
            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(settings.DEVICE), masks.to(settings.DEVICE)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

            # Вывод
            if val_loss is not None:
                print(f"Epoch {epoch_i}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                epoch_time_spent = time.time() - epoch_start

                if avg_sec_per_epoch == 0:
                    avg_sec_per_epoch = epoch_time_spent
                else:
                    avg_sec_per_epoch = (avg_sec_per_epoch * (epoch_i - 1) + epoch_time_spent) / epoch_i

                now = dt.datetime.now()
                sec_left = (avg_sec_per_epoch * (EPOCHS_COUNT - epoch_i))
                print(
                    (
                        f"Epoch {epoch_i}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} "
                        f"| spent {epoch_time_spent:.2f} s | left approx in {sec_left:.2f} s "
                        f"| ends approx: {(now + dt.timedelta(seconds=sec_left)).time()}"
                    ),
                )

            # Сохранение лучшей модели
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and val_loss is not None and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch_i}")
                    break

        # Восстановление лучшей модели
        if save_best and best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(settings.DEVICE)

        self.is_fitted = True

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE)
            outputs = self.model(images)
            preds = (outputs.squeeze() >= 0).to(torch.float32)
        return preds.cpu()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': settings.NUM_CLASSES,
            'LEARNING_RATE': self.LEARNING_RATE,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_classes = settings.NUM_CLASSES
        self.LEARNING_RATE = checkpoint.get('LEARNING_RATE', self.LEARNING_RATE)
        self.is_fitted = True

    @staticmethod
    def get_model_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_ARTERIES/weights.eth")

    @staticmethod
    def get_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_ARTERIES/metrics.csv")

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        """Отрисовывает оригинал, истинную маску и предсказание на переданные оси."""
        # Денормализованное изображение (H,W,3) в диапазоне [0,1]
        img = self._denormalize(image_tensor)
        # Истинная маска в RGB
        true_rgb = self._mask_to_rgb(mask_tensor)
        # Предсказание
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
        """Преобразует нормализованный тензор (C,H,W) в numpy (H,W,3) в диапазоне [0,1]."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def _mask_to_rgb(self, mask_tensor):
        """Преобразует маску (H,W) с индексами классов в RGB (H,W,3) uint8."""
        print(f'MASK SHAPE: {mask_tensor.shape}')
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb

#    @utils.timer
    def _get_mask_only_for(self, class_nums: tp.List[int], masks: torch.Tensor):
        '''
        Technically masks: torch.Tensor[torch.Tensor[torch.Tensor[torch.Tensor[float]]]]
        '''
        if set(class_nums) - set(settings.CLASS_NAMES):
            raise ValueError(f'Classes {class_nums} has values that are not present in settings.CLASS_NAMES')

        class_masks = []
        for class_num in class_nums:
            class_masks.append(masks == class_num)

        res_mask = class_masks[0]
        for class_mask in class_masks[1:]:
            res_mask |= class_mask

        return res_mask.float()
