from collections import defaultdict

import numpy as np
import pandas as pd


df = pd.read_csv(
    r'C:\Users\KRATOS\Desktop\Учёба\Магистратура\2 семестр\Проектный практикум\pp2-veins\p_value\pvalue.csv',
)

def format_mean_std(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # выборочное стандартное отклонение
    # Округляем до 3 знаков, заменяем точку на запятую
    return f"{mean:.3f} ± {std:.3f}".replace('.', ',')


models_metrics = defaultdict(lambda: {'dice': [], 'iou': [], 'precision': [], 'recall': []})

for _, row in df.iterrows():
    model = row['model']
    for col, value in row.items():
        if col in {'model', 'fold'}:
            continue

        models_metrics[model][col].append(value)


total_df = pd.DataFrame(columns=['model', 'dice', 'iou', 'precision', 'recall'])

for model_name, metrics in models_metrics.items():
    new_row_dict = {'model': model_name}
    for metric_name, values in metrics.items():
        new_row_dict[metric_name] = format_mean_std(values)

    new_row = pd.DataFrame([new_row_dict])

    total_df = pd.concat([total_df, new_row], ignore_index=True)

print(total_df)
total_df.to_csv('mean_std_metrics.csv', index=False)
