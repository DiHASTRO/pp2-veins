import pandas as pd
import numpy as np

weights_matrixes = []
for i in range(1, 4):
    df = pd.read_csv(f'weights_matrix_{i}.csv', index_col=0)
    weights_matrixes.append(df)

template = weights_matrixes[0]
result = pd.DataFrame(index=template.index, columns=template.columns)

for idx in result.index:
    for col in result.columns:
        values = [df.loc[idx, col] for df in weights_matrixes]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        # здесь 3 знака
        result.loc[idx, col] = f"{mean_val:.3f} ± {std_val:.3f}"

print(result)
result.to_csv('weights_mean_std_3dec.csv')
