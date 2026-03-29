import pandas as pd
import torch
from tqdm import tqdm
import common.settings as settings
import common.data_preparation as data_prep
from common.metrics import get_all_metrics

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
from BASELINE_V3Plus.model import DeepLabV3Plus, train_extra_transforms, val_extra_transforms
ModelClass = DeepLabV3Plus

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = False
MODEL_SAVE_PATH = ModelClass.get_model_save_path()
METRICS_SAVE_PATH = ModelClass.get_metrics_save_path()

VISUALIZE = True

# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
train_loader = data_prep.get_train_loader(
    train_extra=train_extra_transforms,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS
)
test_loader = data_prep.get_test_loader(
    extra_transforms=val_extra_transforms,
    batch_size=settings.BATCH_SIZE,
    num_workers=settings.NUM_WORKERS
)

# --- МОДЕЛЬ ---
model = ModelClass()
if not USE_FITTED:
    print("Обучение модели...")
    model.fit(train_loader, save_best=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Модель сохранена в {MODEL_SAVE_PATH}")
else:
    print(f"Загрузка предобученной модели из {MODEL_SAVE_PATH}")
    model.load(MODEL_SAVE_PATH)

# --- ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ ---
print("Выполнение предсказаний...")
all_preds = []
all_targets = []

# Добавили tqdm для отображения прогресса по батчам теста
for images, masks in tqdm(test_loader, desc="Inference", unit="batch"):
    preds = model.predict(images)  # (B, H, W) long
    all_preds.append(preds.view(-1))
    all_targets.append(masks.view(-1))

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# --- МЕТРИКИ ---
metrics = get_all_metrics(all_preds, all_targets, num_classes=5)
df = pd.DataFrame([metrics])
df.to_csv(METRICS_SAVE_PATH, index=False)
print(f"Метрики сохранены в {METRICS_SAVE_PATH}")
print("Готово.")