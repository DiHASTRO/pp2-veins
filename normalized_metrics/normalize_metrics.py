import numpy as np
import typing as tp
import sys
import pathlib

SRC_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, SRC_DIR.as_posix())

import pandas as pd


models_metrics_filenames = [
    'DL-CE.csv',
    'DL-DF.csv',
    'DL-WCD.csv',
    'DL-T.csv',
    'TFFM-CE.csv',
    'UN-DF.csv',
    'UN-WCD.csv',
    'ASSEMBLE.csv',
]


classes_folds_counts = {
    'background': [4749964, 4734619, 4751080],
    'arteries': [219114, 218939, 215511],
    'veins': [262023, 274575, 263820],
    'crossings': [9394, 10722, 9182],
    'capillaries': [2385, 4025, 3287],
}


def extract_metrics(df: pd.DataFrame) -> tp.Dict[str, tp.Dict[str, tp.List[float]]]:
    class_names = sorted({col.split('_')[0] for col in df.columns if '_' in col})
    
    return {
        class_name: {
            col.split('_')[1]: list(df[col])
            for col in [col for col in df.columns if class_name in col]
        }
        for class_name in class_names
    }


total_df = pd.DataFrame(columns=['model', 'fold', 'iou', 'dice', 'precision', 'recall'])

for metric_filename in models_metrics_filenames:
    print(f'File: {metric_filename}')
    df = pd.read_csv(metric_filename)
    classes_metrics = extract_metrics(df)

    middlecalc_classes_metrics = {}
    for class_name, metrics in classes_metrics.items():
        print(f'Class name: {class_name}')
        if class_name == 'background':
            print('Skip')
            continue
        
        P = np.array(metrics['precision'])
        R = np.array(metrics['recall'])
        N = np.array(classes_folds_counts[class_name])

        TP = R * N
        FN = N - TP
        FP = np.where(P != 0, TP * (1 - P) / P, 0)
        middlecalc_classes_metrics[class_name] = {
            'TP': np.round(TP),
            'FN': np.round(FN),
            'FP': np.round(FP),
            'P': P,
            'R': R,
            'N': N,
        }

    print(middlecalc_classes_metrics)
    TP_micro = sum(
        metrics['TP']
        for metrics in middlecalc_classes_metrics.values()
    )
    FP_micro = sum(
        metrics['FP']
        for metrics in middlecalc_classes_metrics.values()
    )
    FN_micro = sum(
        metrics['FN']
        for metrics in middlecalc_classes_metrics.values()
    )

    P_micro = TP_micro / (TP_micro + FP_micro)
    R_micro = TP_micro / (TP_micro + FN_micro)

    for i in range(len(P_micro)):
        new_row = pd.DataFrame([{
            'model': metric_filename.split('.')[0],
            'fold': i + 1,
            'iou': df['iou'][i],
            'dice': df['dice'][i],
            'precision': P_micro[i],
            'recall': R_micro[i],
        }])
        total_df = pd.concat([total_df, new_row], ignore_index=True)

total_df.to_csv('real_metrics.csv', index=False)
