import sys
from pathlib import Path
import io

import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_option('client.showErrorDetails', True)

# настройка путей
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from IMPROVED_ARTEM_UNETPP.model import ImprovedUNetPlusPlus, val_extra_transforms
import common.settings as settings

MODEL_PATH = ImprovedUNetPlusPlus.get_model_save_path(1)

@st.cache_resource
def load_model():
    model = ImprovedUNetPlusPlus()
    model.load(MODEL_PATH)
    if hasattr(model, 'eval'):
        model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Те же трансформации, что и в тестовом загрузчике."""
    img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    transform = A.Compose([*val_extra_transforms, ToTensorV2()])
    transformed = transform(image=img_np)
    return transformed["image"]  # [C, H, W]

def get_prediction_mask_only(model, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Использует метод visualize_sample с тремя осями (как в run_pipeline).
    Возвращает RGB-изображение предсказанной маски (без осей).
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    # фиктивная ground truth маска
    dummy_mask = torch.zeros(img_tensor.shape[1:], dtype=torch.long)
    # вызов точно как в исходном коде
    model.visualize_sample(img_tensor, dummy_mask, axes[0], axes[1], axes[2])
    # скрываем оси
    for ax in axes:
        ax.axis('off')
    fig.tight_layout(pad=0)
    # рендерим в буфер
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    full_img = Image.open(buf)
    full_np = np.array(full_img.convert("RGB"))
    plt.close(fig)
    # ширина каждой из трёх колонок
    h, w, _ = full_np.shape
    w_section = w // 3
    # берём правую треть (предсказание)
    pred_np = full_np[:, 2 * w_section:, :]
    return pred_np

def overlay_images(base: np.ndarray, overlay: np.ndarray, alpha=0.4) -> np.ndarray:
    return (base * (1 - alpha) + overlay * alpha).astype(np.uint8)

# ---------- Streamlit UI ----------
st.title("Сегментация сосудов глазного дна")
st.write("Модель выполняет сегментацию сосудов глазного дна и накладывает полученную маску на исходное изображение.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # 1. Исходное изображение
        image = Image.open(uploaded_file)
        orig_np = np.array(image.convert("RGB"))

        # 2. Тензор с трансформациями
        img_tensor = preprocess_image(image)

        # 3. Модель и получение маски
        model = load_model()
        mask_vis = get_prediction_mask_only(model, img_tensor)

        # 4. Приводим маску к размеру оригинала (на случай изменения размера трансформациями)
        mask_pil = Image.fromarray(mask_vis).resize((orig_np.shape[1], orig_np.shape[0]), Image.NEAREST)
        mask_resized = np.array(mask_pil)

        # 5. Наложение
        result = overlay_images(orig_np, mask_resized, alpha=0.4)

        # 6. Вывод
        st.image(result, caption="Исходное изображение + предсказанная маска", use_container_width=True)

    except Exception as e:
        print(e.with_traceback())
        st.error(f"Ошибка при обработке: {e}")
else:
    st.info("Пожалуйста, загрузите изображение.")