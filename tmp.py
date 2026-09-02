import os
import pandas as pd

os.system('cls')

FILE_NAME = 'assemble/raw_3.csv'
METRICS = ['dice', 'iou', 'precision', 'recall']


for metric in METRICS:
    print(metric)
    df = pd.read_csv(FILE_NAME)
    columns_to_account = [col for col in df.columns if metric in col]

    col_sum = sum([df[col] for col in columns_to_account])
    print(col_sum / len(columns_to_account))

exit()

# TOTAL_FILE_NAME = 'p_value/pvalue.csv'
# p_df = pd.read_csv(TOTAL_FILE_NAME)

# print(list(p_df['model'].unique()))
import pandas as pd
import numpy as np
from scipy import stats

# Загрузка данных
df = pd.read_csv('p_value/pvalue.csv')

# Метрики, для которых считаем ДИ
metrics = ['iou', 'dice', 'precision', 'recall']

# Желаемый порядок моделей
model_order = ['DL-CE', 'DL-DF', 'DL-WCD', 'DL-T', 'TFFM-CE', 
               'UN-DF', 'UN-WCD']

# Функция расчёта доверительного интервала (t-распределение)
def compute_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    if n > 1:
        sem = stats.sem(data, ddof=1)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_crit * sem
    else:
        margin = 0.0
    return mean, mean - margin, mean + margin

# Собираем результаты
results = []

# Группируем по модели (можно итерировать по model_order, чтобы сразу сохранить порядок,
# но проще сделать категориальную сортировку позже)
for model, group in df.groupby('model'):
    for metric in metrics:
        values = group[metric].values
        mean, low, high = compute_ci(values)
        results.append({
            'model': model,
            'metric': metric,
            'mean': mean,
            'lower_ci': low,
            'upper_ci': high
        })

# Создаём DataFrame
results_df = pd.DataFrame(results)

# Преобразуем 'model' в категорию с заданным порядком и сортируем
results_df['model'] = pd.Categorical(results_df['model'], categories=model_order, ordered=True)
results_df = results_df.sort_values(['model', 'metric'])

# Выводим на экран
print("95% доверительные интервалы для каждой модели и метрики:\n")
print(results_df.round(6).to_string(index=False))

# Сохраняем в CSV
results_df.to_csv('ci_results.csv', index=False)
print("\nРезультаты сохранены в 'ci_results.csv'")


import pandas as pd
import numpy as np
from scipy import stats

# Загрузка данных
df = pd.read_csv('p_value/pvalue.csv')

# Метрики
metrics = ['iou', 'dice', 'precision', 'recall']


# Функция расчёта 95% ДИ
def compute_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    if n > 1:
        sem = stats.sem(data, ddof=1)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_crit * sem
    else:
        margin = 0.0
    return mean, mean - margin, mean + margin

# Для каждой метрики создаём отдельный DataFrame и сохраняем в CSV
for metric in metrics:
    rows = []
    for model in model_order:
        # Проверяем, есть ли модель в данных (на случай, если какой-то нет)
        if model not in df['model'].values:
            continue
        values = df[df['model'] == model][metric].values
        mean, lower, upper = compute_ci(values)
        rows.append({
            'model': model,
            'mean': mean,
            'lower_ci': lower,
            'upper_ci': upper
        })
    metric_df = pd.DataFrame(rows)
    # Сортируем по model_order (уже порядок сохранён, но на всякий случай)
    metric_df['model'] = pd.Categorical(metric_df['model'], categories=model_order, ordered=True)
    metric_df = metric_df.sort_values('model')
    # Сохраняем
    filename = f'ci_{metric}.csv'
    metric_df.to_csv(filename, index=False)
    print(f"Сохранён {filename}")
    print(metric_df.round(6).to_string(index=False))
    print("\n")



import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('p_value/pvalue.csv')

# Порядок моделей
model_order = ['DL-CE', 'DL-DF', 'DL-WCD', 'DL-T', 'TFFM-CE', 
               'UN-DF', 'UN-WCD']

# Метрики
metrics = ['iou', 'dice', 'precision', 'recall']

# Функция форматирования: mean ± std с заменой точки на запятую
def format_mean_std(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # выборочное стандартное отклонение
    # Округляем до 3 знаков, заменяем точку на запятую
    return f"{mean:.3f} ± {std:.3f}".replace('.', ',')

# Собираем результаты в DataFrame для удобного вывода
table_data = []
for model in model_order:
    if model not in df['model'].values:
        continue
    row = {'Модель': model}
    for metric in metrics:
        values = df[df['model'] == model][metric].values
        row[metric.upper()] = format_mean_std(values)
    table_data.append(row)

result_df = pd.DataFrame(table_data)

# Переименуем колонки для красоты (как в примере)
result_df.rename(columns={'IOU': 'IoU', 'DICE': 'DICE', 
                          'PRECISION': 'Precision', 'RECALL': 'Recall'}, inplace=True)

# Вывод на экран
print("Метрики в формате mean ± std (по трём фолдам):\n")
print(result_df.to_string(index=False))
result_df.to_csv('metrics_mean_std.csv')
