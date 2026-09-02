import sys
import pathlib

SRC_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, SRC_DIR.as_posix())

import typing as tp

from common.base_model import BaseModel
import time

import pandas as pd
import torch
import common.settings as settings
from common import utils
import common.data_preparation as data_prep
from common import metrics

import numpy as np

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
# Импортируем модуль модели и подставляем его в переменную ModelClass
# from UNETPP_VEINS.model import UnetPlusPlus, train_extra_transforms, val_extra_transforms
# from UNETPP_ARTERIES.model import UnetPlusPlus, train_extra_transforms, val_extra_transforms
from ADVANCED_SEGM_V3Plus import model as dl_wcd_module
from UNETPP_EFFB3 import model as un_wcd_module
from TFFM import model as tffm_ce_module


TRAIN_IMAGES_GROUPS = 10

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = True               # False – обучить, True – загрузить готовую
USE_WEIGHTED = True
INTERVAL_METRICS_SAVE_PATH = 'assemble/interval_{i}.csv'
RAW_METRICS_SAVE_PATH = 'assemble/raw_{i}.csv'

VISUALIZE = True  # Показывать ли для визуального сравнения реальные данные и что предсказала модель

def _get_preds_from_both(art_preds, vein_preds):
    art_preds = art_preds.view(-1)
    vein_preds = vein_preds.view(-1)
    new_cells = []
    for cell_1, cell_2 in zip(art_preds, vein_preds):
        new_cells.append(float(cell_1) + float(cell_2))
    return torch.tensor(new_cells)


def ensemble_predict_flat(np_preds, weights_dict):
    """
    Параметры:
        np_preds: dict {model_name: np.ndarray (total_pixels,)} — предсказания моделей
        weights_dict: dict {model_name: list of 5 floats} — веса для классов 0..4
    Возвращает:
        final_pred: np.ndarray (total_pixels,) — итоговая маска после взвешенного голосования
    """
    total_pixels = len(next(iter(np_preds.values())))
    n_classes = 5
    scores = np.zeros((total_pixels, n_classes), dtype=np.float32)
    
    for model_name, pred in np_preds.items():
        weights = weights_dict[model_name]  # список из 5 чисел
        for c in range(n_classes):
            mask = (pred == c)
            scores[mask, c] += weights[c]
    
    final_pred = np.argmax(scores, axis=1)
    return final_pred


def dice_per_class_flat(pred_flat, target_flat, num_classes=5, eps=1e-7):
    """
    pred_flat, target_flat: одномерные np.array одинаковой длины,
                            значения 0..num_classes-1
    Возвращает: np.array dice[class] размера num_classes
    """
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        true_c = (target_flat == c)
        tp[c] = np.sum(pred_c & true_c)
        fp[c] = np.sum(pred_c & ~true_c)
        fn[c] = np.sum(~pred_c & true_c)
    
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice


# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
loaders = data_prep.create_cross_val_loaders(
    train_extra_transforms=dl_wcd_module.train_extra_transforms,
    val_extra_transforms=dl_wcd_module.val_extra_transforms,
)

# --- МОДЕЛЬ ---
assemblies: tp.List[tp.Dict[str, BaseModel]] = [
    {
        'dl_wcd': dl_wcd_module.ConsistentDeepLabV3Plus(),
        'un_wcd': un_wcd_module.UNetPlusPlusEffB3(),
        'tffm_ce': tffm_ce_module.TFFMModel(),
    },
    {
        'dl_wcd': dl_wcd_module.ConsistentDeepLabV3Plus(),
        'un_wcd': un_wcd_module.UNetPlusPlusEffB3(),
        'tffm_ce': tffm_ce_module.TFFMModel(),
    },
    {
        'dl_wcd': dl_wcd_module.ConsistentDeepLabV3Plus(),
        'un_wcd': un_wcd_module.UNetPlusPlusEffB3(),
        'tffm_ce': tffm_ce_module.TFFMModel(),
    },
]

for i, models in enumerate(assemblies, start=1):
    print(f'Загружаем фолд {i}')
    for name, model in models.items():
        save_path = model.get_model_save_path(i)
        print(f"Загрузка предобученной модели {name} из {save_path}")
        model.load(save_path)


if not USE_WEIGHTED:
    weights_dfs = []
    for assembly_i, (assembly, loader) in enumerate(zip(assemblies, loaders), start=1):
        print(f'Ансамбль {assembly_i}')
        start = time.time()
        train_loader, val_loader = loader

        models_preds = {
            name: []
            for name in assembly.keys()
        }
        targets = []

        for i, (images, masks) in enumerate(train_loader, start=1):
            print(f'Изображение {i}')
            for name, model in assembly.items():
                print(f'Сейчас предсказывает модель {name} ...')
                models_preds[name].append(model.predict(images).view(-1))

            targets.append(masks.view(-1))
            now = time.time()
            print(f'Прошло {utils.beautify_time_left(now - start)}. Осталось {utils.beautify_time_left((now - start) / i * (TRAIN_IMAGES_GROUPS - i))}')

        np_preds = {
            model_name: torch.cat(preds).numpy()
            for model_name, preds in models_preds.items()
        }
        np_targets = torch.cat(targets).numpy()

        model_names = list(np_preds.keys())
        num_classes = 5

        weights_dict = {}
        for model_name in model_names:
            print(f'Вычисляем метрики для {model_name} ...')
            pred_flat = np_preds[model_name]
            dice_scores = dice_per_class_flat(pred_flat, np_targets, num_classes)
            weights_dict[model_name] = dice_scores

        weights_df = pd.DataFrame(weights_dict, index=[f'class_{c}' for c in range(num_classes)]).T
        print("Поклассовые Dice (веса для голосования):")
        print(weights_df.round(4))

        # Матрица весов (модели x классы)
        weights_matrix = np.array([weights_dict[m] for m in model_names])
        print("\nМатрица весов:\n", weights_matrix.round(4))
        weights_dfs.append(weights_df)
        weights_df.to_csv(f'weights_matrix_{assembly_i}.csv')
else:
    weights_dfs = [
        pd.read_csv(f'weights_matrix_{i}.csv', index_col=0)
        for i in range(1, 4)
    ]

print(weights_dfs)

root_power = 10
print(
    [
        {
            'dl_wcd': np.array(weights_df.loc['dl_wcd'].tolist()) ** (1 / root_power),
            'un_wcd': np.array(weights_df.loc['un_wcd'].tolist()) ** (1 / root_power),
            'tffm_ce': np.array(weights_df.loc['tffm_ce'].tolist()) ** (1 / root_power),
        }
        for weights_df in weights_dfs
    ]
)

start = time.time()
for root_power in range(1):
    folds_models_weights = [
        {
            'dl_wcd': [1, 1, 1, 1, 1],
            'un_wcd': [1, 1, 1, 1, 1],
            'tffm_ce': [1, 1, 1, 1, 1],
        }
        for weights_df in weights_dfs
    ]

    # --- ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ ---
    print("Выполнение предсказаний...")
    raw_metrics = []
    for i, (assembly, loader, models_weights) in enumerate(zip(assemblies, loaders, folds_models_weights), start=1):
        print(f'Fold {i} ...')

        _, val_loader = loader
        models_preds = {
            name: []
            for name in assembly.keys()
        }
        targets = []
        for i, (images, masks) in enumerate(val_loader, start=1):
            print(f'Изображение {i}')
            for name, model in assembly.items():
                print(f'Сейчас предсказывает модель {name} ...')
                models_preds[name].append(model.predict(images).view(-1))

            targets.append(masks.view(-1))
            now = time.time()

        np_preds = {
            model_name: torch.cat(preds).numpy()
            for model_name, preds in models_preds.items()
        }
        np_targets = torch.cat(targets).numpy()

        final_predictions = ensemble_predict_flat(np_preds, models_weights)
        raw_metrics.append(metrics.get_all_metrics(final_predictions, np_targets, num_classes=5))

    raw_df = pd.DataFrame(raw_metrics)
    raw_df.to_csv(RAW_METRICS_SAVE_PATH.format(i='EQUALS'), index=False)

    interval_df = metrics.get_interval_metrics_from_raw(raw_df)
    interval_df.to_csv(INTERVAL_METRICS_SAVE_PATH.format(i='EQUALS'), index=False)
    print(f"Сырые метрики сохранены в {RAW_METRICS_SAVE_PATH.format(i='EQUALS')}")
    print(f"Интервальные оценки сохранены в {INTERVAL_METRICS_SAVE_PATH.format(i='EQUALS')}")
    
    now = time.time()
    print(f'Прошло {utils.beautify_time_left(now - start)}. Осталось {utils.beautify_time_left((now - start) / root_power * (10 - root_power))}')
