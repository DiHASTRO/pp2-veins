# import numpy as np
# from PIL import Image
# from albumentations.pytorch import ToTensorV2
# import albumentations as A

# from common import settings

# from BASELINE_V3Plus import model as dl_ce
# model = dl_ce.DeepLabV3Plus()


# def mask_to_rgb(mask_tensor):
#     """Преобразует маску (H,W) с индексами классов в RGB (H,W,3) uint8."""
#     mask = mask_tensor.cpu().numpy().astype(np.uint8)
#     h, w = mask.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     for cls, color in settings.COLOR_MAP.items():
#         rgb[mask == cls] = color
#     return rgb


# NUM_CLASSES = 5
# IMG_SIZE = 512
# EPOCHS_COUNT = 50
# LEARNING_RATE = 1e-4

# # Константы для модели
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)

# transforms = [
#     A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
#     A.Normalize(mean=MEAN, std=STD),
#     ToTensorV2(),
# ]

import numpy as np
from PIL import Image

# source = np.array(Image.open('dataset/images/015_N.png').convert('RGB'))
true_mask = np.array(Image.open('dataset/masks/015_N.png').convert('RGB'))


# source_part = source[part_y_description, part_x_description]

# augmented = A.Compose(transforms)(image=source_part, mask=true_mask_part)
# source_part_prepared = augmented['image']
# mask_part_prepared = augmented['mask']

# result = model.predict(source_part_prepared.unsqueeze(0)).squeeze(0).cpu()
# rgb = mask_to_rgb(result)
# Image.fromarray(rgb).show()

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from UNETPP_EFFB3.model import UNetPlusPlusEffB3, MEAN, STD, IMG_SIZE
import common.settings as settings

# Загружаем модель (веса первого фолда)
model = UNetPlusPlusEffB3()
model.load(UNetPlusPlusEffB3.get_model_save_path(1))
model.model.eval()

# Определяем устройство модели (CPU или CUDA)
device = next(model.model.parameters()).device
print(f"Модель на устройстве: {device}")

# Загружаем и преобразуем изображение
image = np.array(Image.open('dataset/images/015_N.png').convert('RGB'))
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
input_tensor = transform(image=image)['image'].unsqueeze(0)

# Перемещаем тензор на то же устройство, где модель
input_tensor = input_tensor.to(device)

# Предсказываем маску
with torch.no_grad():
    logits = model.predict(input_tensor)
    mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

# Преобразуем маску в цветное RGB
h, w = mask.shape
rgb = np.zeros((h, w, 3), dtype=np.uint8)
for cls, color in settings.COLOR_MAP.items():
    rgb[mask == cls] = color

# Показываем маску
Image.fromarray(rgb).save('UN-WCD-part.png')

