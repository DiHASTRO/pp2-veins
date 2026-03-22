# pp2-veins
Пока без форматирования особого...

Чтобы начать работать, делаем

```
pip install -r requirements.txt
```

Далее создаём свою папку прямо в корне по аналогии с BASELINE_V3Plus, там файлs
- model.py
- __init__.py

В model.py создаём класс своей модели и наследуем его от BaseModel

Будет что-то типа

```py
from common.base_model import BaseModel

class MyNewMegaModel(BaseModel):
    ...
```

Далее мы реализуем все абстрактные методы из BaseModel. Иначе дальнейший запуск просто упадёт с ошибкой.

Далее переходим в файл run_pipeline.py

Импортируем свою модель вместо BASELINE_V3Plus и делаем ModelClass = MyNewMegaModel.

USE_FITTED - если True, то мы подгружаем из указанного в модели пути файл с весами. Иначе обучаем.

Ваша модель обучится, метрики сложатся в файл, который вы так же определите в классе модели.

Что важно знать:

1. В модели делаются базовые преобразования: аугментация, плюс приведение к TensorV2:

```py
A.RandomRotate90(p=0.5),
A.HorizontalFlip(p=0.5),
EXTRAS
ToTensorV2()
```

Собственно так как каждая модель может требовать какие-то специфичные преобразования, EXTRAS можно подменить на нужные преобразования.
Для этого указываем в файле модели переменную что-то типа

```py
train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]
```

Итого наш пайплайн загрузки данных превратится в

```py
A.RandomRotate90(p=0.5),
A.HorizontalFlip(p=0.5),
A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
A.Normalize(mean=MEAN, std=STD),
ToTensorV2()
```

TBD
