from PIL import Image
import numpy as np
from pathlib import Path

# возьми любую маску из нарезанного датасета
mask_path = sorted(Path("dataset_patches_256_overlap/masks").glob("*.png"))[0]
print("Файл:", mask_path)

# PIL читает как RGB — так же, как в SegmentationDataset
mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
print("shape:", mask_rgb.shape, "dtype:", mask_rgb.dtype)

# уникальные цвета
pixels = mask_rgb.reshape(-1, 3)
colors, counts = np.unique(pixels, axis=0, return_counts=True)
order = np.argsort(-counts)

print("\nЦвета (в порядке RGB, как их видит PIL):")
for i in order:
    r, g, b = colors[i].tolist()
    print(f"  RGB=({r:3d},{g:3d},{b:3d})  count={counts[i]:>10}")

from common import settings

print("\nCOLOR_MAP из settings:")
for cls, color in settings.COLOR_MAP.items():
    print(f"  class {cls}: {color}")

print("\nПроверка: какие цвета из COLOR_MAP есть в маске (RGB):")
present = set(map(tuple, colors.tolist()))
for cls, color in settings.COLOR_MAP.items():
    ok = tuple(color) in present
    print(f"  class {cls} {color}: {'OK' if ok else 'НЕ НАЙДЕН в маске'}")

print("\nЦвета, которые есть в маске, но не в COLOR_MAP:")
mapped = {tuple(c) for c in settings.COLOR_MAP.values()}
for c in present:
    if c not in mapped:
        print(f"  {c}")