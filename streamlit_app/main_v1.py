import sys
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------- импорты из проекта ----------------------
root_dir = Path(__file__).resolve().parent.parent   # корень проекта
sys.path.insert(0, str(root_dir))

from BASELINE_V3Plus.model import DeepLabV3Plus, val_extra_transforms
import common.settings as settings

MODEL_PATH = DeepLabV3Plus.get_model_save_path(1)   # путь к весам (fold 1)

@st.cache_resource
def load_model():
    """Загружаем и кешируем модель."""
    model = DeepLabV3Plus()
    model.load(MODEL_PATH)
    # Приводим в режим оценки, если метод существует (не у всех моделей)
    if hasattr(model, 'eval'):
        model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Применяем те же трансформации, что и в тестовом загрузчике:
    val_extra_transforms + ToTensorV2.
    Возвращает тензор [C, H, W] на CPU.
    """
    img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    transform = A.Compose([*val_extra_transforms, ToTensorV2()])
    transformed = transform(image=img_np)
    return transformed["image"]   # [C, H, W], float

# ---------------------- UI ----------------------
st.title("Сегментация изображений – DeepLabV3+")
st.write("Загрузите снимок, и модель построит маску, используя тот же метод визуализации, что и в `run_pipeline.py`.")

uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # 1. читаем файл
        image = Image.open(uploaded_file)

        # 2. препроцессинг (как в test_loader)
        img_tensor = preprocess_image(image)

        # 3. загружаем модель
        model = load_model()

        # 4. создаём такую же фигуру, как в run_pipeline.py (1 строка, 3 столбца)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Для метода visualize_sample нужна маска – дадим фиктивный тензор нужного размера
        # (ground truth в приложении отсутствует, поэтому показываем только оригинал и предсказание)
        dummy_mask = torch.zeros(img_tensor.shape[1:], dtype=torch.long)  # [H, W]

        # 5. вызываем в точности тот же метод, что и в коде визуализации
        model.visualize_sample(img_tensor, dummy_mask, axes[0], axes[1], axes[2])

        # 6. скрываем среднюю ось (истинная маска) – оставляем исходное изображение и предсказание
        axes[1].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка при обработке: {e}")
else:
    st.info("Пожалуйста, загрузите изображение.")
