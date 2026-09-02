import pandas as pd
import torch
import common.settings as settings
import common.data_preparation as data_prep
from common import metrics

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
# Импортируем модуль модели и подставляем его в переменную ModelClass
# from UNETPP_VEINS.model import UnetPlusPlus, train_extra_transforms, val_extra_transforms
# from UNETPP_ARTERIES.model import UnetPlusPlus, train_extra_transforms, val_extra_transforms
from UNETPP_VEINS import model as veins_model
from UNETPP_ARTERIES import model as arteries_model
ArteriesModelClass = arteries_model.UnetPlusPlus
VeinsModelClass = veins_model.UnetPlusPlus

import pathlib

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = True               # False – обучить, True – загрузить готовую
INTERVAL_METRICS_SAVE_PATH = pathlib.Path('interval.csv')
RAW_METRICS_SAVE_PATH = pathlib.Path('raw.csv')

VISUALIZE = True  # Показывать ли для визуального сравнения реальные данные и что предсказала модель

def _get_preds_from_both(art_preds, vein_preds):
    art_preds = art_preds.view(-1)
    vein_preds = vein_preds.view(-1)
    new_cells = []
    for cell_1, cell_2 in zip(art_preds, vein_preds):
        new_cells.append(float(cell_1) + float(cell_2))
    return torch.tensor(new_cells)


# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
loaders = data_prep.create_cross_val_loaders(
    train_extra_transforms=veins_model.train_extra_transforms,
    val_extra_transforms=veins_model.val_extra_transforms,
)

# --- МОДЕЛЬ ---
models = [(ArteriesModelClass(), VeinsModelClass()) for _ in range(settings.FOLDS_NUM)]
if not USE_FITTED:
    print("Обучение модели...")
    for i, (model, (train_loader, val_loader)) in enumerate(zip(models, loaders), start=1):
        print(f'Бакет {i}...')
        model.fit(train_loader, save_best=True)

        model_save_path = ModelClass.get_model_save_path(i)
        model.save(model_save_path)
        print(f"Модель сохранена в {model_save_path}")
else:
    for i, (arteries_model_impl, veins_model_impl) in enumerate(models, start=1):
        art_model_save_path = arteries_model_impl.get_model_save_path(i)
        vein_model_save_path = veins_model_impl.get_model_save_path(i)

        print(f"Загрузка предобученной модели из {art_model_save_path}, {vein_model_save_path}")
        arteries_model_impl.load(art_model_save_path)
        veins_model_impl.load(vein_model_save_path)

# --- ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ ---
print("Выполнение предсказаний...")
raw_metrics = []
for i, ((art_model, vein_model), test_loader) in enumerate(zip(models, [loader[1] for loader in loaders]), start=1):
    print(f'Для модели {i} ...')
    all_preds = []
    all_targets = []
    for images, masks in test_loader:
        art_preds = art_model.predict(images)  # (B, H, W) long
        vein_preds = vein_model.predict(images)  # (B, H, W) long

        preds = _get_preds_from_both(art_preds, vein_preds)
        all_preds.append(preds)
        all_targets.append(masks.view(-1))

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    raw_metrics.append(metrics.get_all_metrics(all_preds, all_targets, num_classes=5))

raw_df = pd.DataFrame(raw_metrics)
raw_df.to_csv(RAW_METRICS_SAVE_PATH, index=False)

interval_df = metrics.get_interval_metrics_from_raw(raw_df)
interval_df.to_csv(INTERVAL_METRICS_SAVE_PATH, index=False)
print(f"Сырые метрики сохранены в {RAW_METRICS_SAVE_PATH}")
print(f"Интервальные оценки сохранены в {INTERVAL_METRICS_SAVE_PATH}")
print("Готово.")
