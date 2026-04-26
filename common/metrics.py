import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from common import settings


def compute_macro_dice_iou(cm):
    """Возвращает macro average Dice и IoU по всем классам."""
    num_classes = cm.shape[0]
    dice_per_class = []
    iou_per_class = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice_per_class.append(dice)
        iou_per_class.append(iou)
    return np.mean(dice_per_class), np.mean(iou_per_class)


def compute_per_class_accuracy_precision_recall(cm, class_names):
    """
    Возвращает per-class accuracy, precision, recall.
    Ключи: f'{class_name}_accuracy', f'{class_name}_precision', f'{class_name}_recall'.
    """
    num_classes = cm.shape[0]
    result = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        class_name = class_names.get(cls, f'class_{cls}')
        result[f'{class_name}_accuracy'] = accuracy
        result[f'{class_name}_precision'] = precision
        result[f'{class_name}_recall'] = recall
    return result


def get_all_metrics(preds, targets, num_classes, class_names=None):
    """
    Возвращает общие метрики, per-class метрики и специальные artery/vein error rates.
    Error rates нужны для проверки гипотезы про снижение путаницы artery <-> vein.
    """
    if class_names is None:
        class_names = settings.CLASS_NAMES

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    macro_dice, macro_iou = compute_macro_dice_iou(cm)
    per_class_metrics = compute_per_class_accuracy_precision_recall(cm, class_names)

    artery_to_vein = cm[1, 2] / (cm[1, :].sum() + 1e-8)
    vein_to_artery = cm[2, 1] / (cm[2, :].sum() + 1e-8)

    return {
        'dice': macro_dice,
        'iou': macro_iou,
        'artery_to_vein_rate': artery_to_vein,
        'vein_to_artery_rate': vein_to_artery,
        **per_class_metrics,
    }


def get_interval_metrics_from_raw(metrics_df: pd.DataFrame) -> pd.DataFrame:
    confidence_level = getattr(settings, 'CONFIDENCE_LEVEL', 0.95)
    alpha = 1 - confidence_level
    n = len(metrics_df)
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    results = []
    for column in metrics_df.columns:
        data = metrics_df[column]
        mean_val = data.mean()
        sem = data.sem()
        margin = t_critical * sem
        results.append({
            'metric_name': column,
            'lower_bound': mean_val - margin,
            'average': mean_val,
            'upper_bound': mean_val + margin,
        })

    return pd.DataFrame(results)
