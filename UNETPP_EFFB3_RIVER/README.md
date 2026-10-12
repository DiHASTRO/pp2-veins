# Сегментация сосудов с attention из речного RAU-Net++

`UNetPlusPlusEffB3River` — самостоятельная модель на основе `UNETPP_EFFB3`.
Она переносит способ сохранения деталей тонких речных ответвлений на сегментацию
сосудов: CBAM в конце encoder и остаточное слияние признаков RAFF на коротких
связях UNet++. Ожидаемый эффект — более точные границы и меньше пропущенных
тонких сосудов. Улучшение на текущих данных требует сравнения после обучения.

## Запуск с нуля в Windows PowerShell

Инструкция рассчитана на Windows 10/11, 64-битный Python и PowerShell.
Все команды ниже выполняются из корня проекта, в примерах это `E:\pp2-veins`.
Если проект расположен в другой папке, замените этот путь на свой.
Выполняйте блоки по порядку; при ошибке сначала устраните её.

Нужны папка проекта с `common`, `UNETPP_EFFB3_RIVER` и `run_pipeline_river.py`,
а также размеченный датасет. Установленный Python и готовый `.venv` заранее
не требуются. Для установки библиотек и первой загрузки ImageNet weights нужен
интернет. Для обучения рекомендуется NVIDIA GPU; вариант с CPU приведён ниже.
Ручной запуск проверен на RTX 3060 с 12 ГиБ памяти.

### 1 Установите Python

Откройте [страницу Python 3.12.10](https://www.python.org/downloads/release/python-31210/),
найдите раздел Files и скачайте **Windows installer (64-bit)**.
Это версия Python, использованная для проверки модели.

Запустите установщик, включите **Add python.exe to PATH** и установку launcher
`py`, затем выберите **Install Now**. После установки откройте новое окно
PowerShell и проверьте:

```powershell
py -3.12 --version
```

Ожидается `Python 3.12.10`. Если команда `py` не найдена, закройте и снова
откройте PowerShell. Если это не помогло, повторно запустите установщик Python
и включите Python Launcher.

### 2 Откройте папку проекта

Получите папку проекта с кодом и данными, например распакуйте предоставленный
архив в `E:\pp2-veins`. Затем выполните:

```powershell
Set-Location -LiteralPath 'E:\pp2-veins'
Get-Location
Test-Path -LiteralPath '.\run_pipeline_river.py'
Test-Path -LiteralPath '.\UNETPP_EFFB3_RIVER\requirements.txt'
```

Две последние команды должны вывести `True`. Не запускайте обучение из папки
`UNETPP_EFFB3_RIVER`: относительные пути результатов рассчитаны на корень проекта.

### 3 Создайте виртуальное окружение

`.venv` — отдельная папка с Python и библиотеками этого проекта.
Если её ещё нет, создайте её:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print('venv:', sys.prefix != sys.base_prefix)"
```

Ожидается путь `E:\pp2-veins\.venv\Scripts\python.exe` и `venv: True`.
Если `.venv` уже существует, работает и использует Python 3.12, создание можно
пропустить. Если в нём другой Python, сохраните старое окружение и создайте
новое командой `py -3.12 -m venv .venv-river`. В этом случае во всех дальнейших
командах замените `.venv` на `.venv-river`.

Дальнейшие команды прямо вызывают Python из `.venv`, поэтому активация
не обязательна. Такой способ описан в [документации Python venv](https://docs.python.org/3.12/library/venv.html)
и работает независимо от разрешения запускать `Activate.ps1`.

### 4 Установите PyTorch

Сначала обновите pip внутри созданного окружения:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Выберите **один** из двух вариантов установки PyTorch.
Команды соответствуют [официальной инструкции для PyTorch 2.7.1](https://pytorch.org/get-started/previous-versions/#v271).

**Вариант A — NVIDIA GPU**, в том числе RTX 3060 из проверенной конфигурации.
Установите актуальный драйвер своей видеокарты с
[сайта NVIDIA](https://www.nvidia.com/en-us/drivers/) и проверьте, что система
её видит:

```powershell
nvidia-smi
```

После этого установите PyTorch с CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

**Вариант B — CPU**, если совместимой NVIDIA GPU нет:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Модель выбирает CPU автоматически, когда CUDA недоступна. Полное обучение
на CPU обычно существенно медленнее.

### 5 Установите остальные зависимости

Используйте файл с версиями, проверенными для новой модели:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\UNETPP_EFFB3_RIVER\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`pip check` должен вывести `No broken requirements found`.
Файл содержит также выбранные версии torch и torchvision: уже установленная
сборка `2.7.1+cu128` или `2.7.1+cpu` удовлетворяет записи `torch==2.7.1`.
Корневой `requirements.txt` менять не требуется.

Проверьте окружение и импорт модели:

```powershell
$env:NO_ALBUMENTATIONS_UPDATE = '1'
.\.venv\Scripts\python.exe -c "import torch, segmentation_models_pytorch as smp; from common import settings; from UNETPP_EFFB3_RIVER.model import UNetPlusPlusEffB3River; print('torch:', torch.__version__); print('SMP:', smp.__version__); print('CUDA:', torch.cuda.is_available()); print('DEVICE:', settings.DEVICE); print('MODEL:', UNetPlusPlusEffB3River.get_model_name())"
```

Ожидаются SMP `0.5.0` и MODEL `UN-WCD-RIVER`. Для варианта NVIDIA ожидаются
`CUDA: True` и `DEVICE: cuda`; для CPU — `CUDA: False` и `DEVICE: cpu`.
`NO_ALBUMENTATIONS_UPDATE` отключает сетевую проверку новых версий библиотеки
в текущем окне PowerShell; преобразования изображений остаются прежними.

### 6 Подготовьте расположение датасета

Используйте существующие изображения и RGB-маски. Структура папок:

```text
E:\pp2-veins\
    run_pipeline_river.py
    common\
    UNETPP_EFFB3_RIVER\
        model.py
        river_attention.py
        requirements.txt
    dataset\
        images\
            001_G.png
            002_N.png
            ...
        masks\
            001_G.png
            002_N.png
            ...
```

Для каждого изображения нужен файл маски с тем же именем. Загрузчик использует
PNG из `dataset/images` и `dataset/masks`; JPG, отдельные папки train/test
или внешние датасеты этот runner не подхватывает. При исходных настройках
`FOLDS_NUM=3` и `TEST_SIZE=20` требуется не менее **60 пар**.

Цвета RGB-маски должны соответствовать `common/settings.py`:

| Класс | RGB |
| --- | --- |
| Фон | `(0, 0, 0)` |
| Артерии | `(255, 0, 0)` |
| Вены | `(0, 0, 255)` |
| Пересечения | `(0, 255, 0)` |
| Капилляры | `(255, 255, 255)` |

Проверьте число файлов и совпадение имён до запуска:

```powershell
.\.venv\Scripts\python.exe -c "from common import settings; images = {p.name for p in settings.DATASET_IMG_DIR.glob('*.png')}; masks = {p.name for p in settings.DATASET_MASK_DIR.glob('*.png')}; print('Images:', len(images)); print('Masks:', len(masks)); print('Required pairs:', settings.FOLDS_NUM * settings.TEST_SIZE); print('Images without mask:', sorted(images - masks)); print('Masks without image:', sorted(masks - images))"
```

Числа Images и Masks должны совпасть и быть не меньше Required pairs.
Оба списка отсутствующих пар должны быть `[]`. В текущем проекте есть 100 пар.
Resize, Normalize, аугментации и создание folds выполняет существующий код;
заранее изменять размеры изображений или переписывать подготовку данных не нужно.

### 7 Запустите обучение и расчёт метрик

В `run_pipeline_river.py` уже выбрана новая модель и установлено
`USE_FITTED = False`. Это копия `run_pipeline_6.py` с заменой только импорта
модели и `ModelClass`; остальные шаги pipeline сохранены.

```powershell
.\.venv\Scripts\python.exe -u .\run_pipeline_river.py
```

Сначала появится сообщение «Загрузка данных...». При первом запуске конструктор
скачает ImageNet weights EfficientNet-B3. После создания моделей появятся
«Обучение модели...», номер бакета и строки `Epoch 1/50`, `Train Loss`.
Для каждого из трёх folds выполняются
50 эпох; после всех folds runner рассчитывает метрики и сохраняет CSV.
Не закрывайте окно PowerShell во время обучения.

Текущий pipeline вызывает `fit(train_loader, save_best=True)` без validation
loader: лучшая эпоха и early stopping здесь не выбираются. Веса сохраняются
после завершения обучения каждого fold. `Ctrl+C` останавливает процесс;
уже завершённые folds сохраняются, незавершённый fold автоматически не возобновляется.

Ожидаемые файлы после успешного завершения:

```text
UNETPP_EFFB3_RIVER\weights_1.eth
UNETPP_EFFB3_RIVER\weights_2.eth
UNETPP_EFFB3_RIVER\weights_3.eth
UNETPP_EFFB3_RIVER\raw_metrics.csv
UNETPP_EFFB3_RIVER\interval_metrics.csv
```

Проверьте результаты:

```powershell
Get-ChildItem -LiteralPath .\UNETPP_EFFB3_RIVER -File
Import-Csv -LiteralPath .\UNETPP_EFFB3_RIVER\raw_metrics.csv | Format-Table
```

Повторный запуск с `USE_FITTED = False` обучает модель заново и перезаписывает
её результаты. Если предыдущие веса и CSV нужны для сравнения, предварительно
сохраните их копию в другой папке.

### 8 Посчитайте метрики по готовым весам

Для этого нужны все три файла `weights_1.eth`, `weights_2.eth` и `weights_3.eth`
именно этой модели, а также тот же датасет и настройки разбиения.

Откройте runner:

```powershell
notepad .\run_pipeline_river.py
```

Измените только строку `USE_FITTED = False` на `USE_FITTED = True`, сохраните
файл и запустите ту же команду:

```powershell
.\.venv\Scripts\python.exe -u .\run_pipeline_river.py
```

В этом режиме runner загружает готовые веса и заново рассчитывает CSV-метрики.
Это оценка готовой модели, а не продолжение прерванного обучения.
Веса из папки исходной `UNETPP_EFFB3` для новой архитектуры не подходят.

### 9 Вернитесь к проекту в новом окне PowerShell

Если окружение и зависимости уже установлены, повторять установку не нужно:

```powershell
Set-Location -LiteralPath 'E:\pp2-veins'
$env:NO_ALBUMENTATIONS_UPDATE = '1'
.\.venv\Scripts\python.exe -u .\run_pipeline_river.py
```

Для удобства можно активировать окружение и писать просто `python`:

```powershell
.\.venv\Scripts\Activate.ps1
python -u .\run_pipeline_river.py
```

Если PowerShell блокирует `Activate.ps1`, продолжайте использовать полный путь
`.\.venv\Scripts\python.exe`, как в основной инструкции. Выход из активированного
окружения — команда `deactivate`; само окружение при этом сохраняется.

### Частые ошибки запуска

| Сообщение или симптом | Что сделать |
| --- | --- |
| `py` не найден или Python 3.12 не установлен | Установите 64-битный Python из шага 1 с launcher и откройте новое окно PowerShell |
| `.venv\Scripts\python.exe` не найден | Проверьте текущую папку через `Get-Location` и создайте `.venv` по шагу 3 |
| `No module named ...` | Выполните установку из шага 5 через Python именно этого `.venv` |
| `No matching distribution found` | Проверьте Python 3.12 и 64-битную архитектуру: `.\.venv\Scripts\python.exe -c "import sys, struct; print(sys.version); print(struct.calcsize('P') * 8)"`; ожидается `64`. Затем проверьте доступ к выбранному pip index |
| `No module named common` или runner не найден | Перейдите в корень проекта и запускайте `run_pipeline_river.py` из него |
| `CUDA: False` при установленной NVIDIA GPU | Проверьте `nvidia-smi` и `torch.__version__`. Если установлен CPU torch, переустановите пару: `.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128`, затем повторите проверки шага 5 |
| `CUDA out of memory` | Закройте другие программы, использующие GPU, и повторите запуск. Для GPU с меньшей памятью можно уменьшить `BATCH_SIZE` в `common/settings.py`, например до 4 или 2; это уже изменение параметров эксперимента, исходное значение — 8 |
| «Всего ... изображений, требуется ...» | Добавьте недостающие PNG-пары и повторите проверку шага 6 |
| Ошибка скачивания pretrained weights | Проверьте интернет, proxy и доступ к `huggingface.co`; первая загрузка ImageNet weights требует подключения |
| `weights_*.eth` не найден при `USE_FITTED = True` | Завершите обучение всех folds или скопируйте все три совместимых файла весов в указанную папку |
| `Checkpoint is not a UNETPP_EFFB3_RIVER model` | Используйте веса новой модели, а не baseline или другого эксперимента |

## Научное основание

Основной источник — Tang, Zhang, Jiang, Lin, Hou,
[RAU-Net++: River Channel Extraction Methods for Remote Sensing Images of Cold and Arid Regions](https://www.mdpi.com/2076-3417/14/1/251),
Applied Sciences, 2024, 14(1), 251, DOI `10.3390/app14010251`
([полный текст](https://artefacts-discovery.researcher.life/full_text/DA-2/1b/1b78cee5a8ad325e8b0e66b1c3f87537/full_text/3bfbfa2c3323bc5cd34be5a1b00169a1.pdf)).
Статья опубликована 27 декабря 2023 года в выпуске 2024 года.
Авторы используют UNet++, VGG16, CBAM и RAFF для бинарной сегментации русел
по снимкам Landsat8. Разделы 3.2.2 и 3.3.2 описывают attention и три ветви RAFF.
В таблице 10 вариант с одним CBAM уступает основе, а сочетание CBAM и RAFF
даёт улучшение. Поэтому в модель перенесены оба модуля.

Канонический CBAM описан в Woo et al.,
[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521),
ECCV 2018. Общий MLP обрабатывает average и maximum pooling по пространству;
за каналами следует пространственная маска с convolution 7×7.

Другие изученные речные подходы:

| Работа | Механизм | Решение для этого эксперимента |
| --- | --- | --- |
| Isikdogan, Bovik, Passalacqua, [Learning a River Network Extractor Using an Adaptive Loss Function](https://doi.org/10.1109/LGRS.2018.2811754), IEEE GRSL, 2018; [PDF авторов](https://www.isikdogan.com/files/Isikdogan2018_deepriver.pdf) | DeepRiver извлекает осевые линии из карт воды; локальный max-pooling CE акцентирует трудные пиксели | Оставлен для отдельного эксперимента с loss |
| Dui et al., [Segmentation and Connectivity Reconstruction of Urban Rivers … by the WaterSCNet Deep Learning Model](https://www.mdpi.com/2072-4292/15/19/4875), Remote Sensing, 2023, DOI `10.3390/rs15194875` | Каскад сегментации и реконструкции связности; отдельные ручные connectivity labels | Требует дополнительной разметки, несовместимой с условием сохранить данные |

Soft-clDice также рассматривался по
[Shit et al., CVPR 2021](https://arxiv.org/abs/2003.07311).
Это работа о сосудах, дорогах и нейронах, а не о реках. В текущий loss она
не добавлена, чтобы сравнение измеряло архитектурный перенос RAU-Net++.

## Реализованная архитектура

RGB → EfficientNet-B3 → CBAM на самом глубоком уровне → штатный UNet++ decoder
с RAFF на входящих коротких связях → штатная segmentation head → 5 logits.

`river_attention.py` сохраняет граф, convolution blocks и interpolation SMP.
Каждая группа признаков внутри объединённого skip проходит свой RAFF перед
повторным concatenation. Получаются 20 RAFF в 10 блоках decoder. Последний блок
без skip остаётся штатным. Все attention-модули зарегистрированы внутри
`self.model` до создания optimizer и входят в checkpoint.

RAFF сохраняет форму признаков и складывает три ветви:

```text
pool(x)       = ReLU(BN(Conv1x1(MaxPool3x3(x))))
bottleneck(x) = ReLU(BN(Conv3x3(ReLU(BN(Conv1x1(x))))))
gate(x)       = sigmoid(Conv1x1(ReLU(Conv1x1(GlobalAvgPool(x)))))
RAFF(x)       = x + pool(x) + bottleneck(x) * gate(x)
```

Это адаптация к EfficientNet-B3 и пяти классам. Порядок операций RAFF выбран по
тексту раздела 3.3.2: его формула (6) расходится с описанием. Размер bottleneck
уменьшается в 4 раза, channel MLP — в 16 раз; это настройки реализации.
MaxPool использует stride=1 и padding=1. BN исключён только из global gate,
чтобы последний тренировочный batch из одного изображения работал корректно.
Специальное объединение pooling с downsampling VGG16 из оригинала не переносится
на EfficientNet-B3.

Обучение использует activation checkpointing encoder и блоков decoder с RAFF:
они пересчитываются при backward вместо хранения всех промежуточных активаций.
Для drop-connect EfficientNet сохраняется RNG state, чтобы повторить исходные
случайные маски. Во время пересчёта BN
получает временные копии своих buffers, поэтому настоящие running statistics
обновляются один раз. При `eval()` и `no_grad()` пересчёт отключён.

## Данные и обучение

Сохранены Resize 512×512, ImageNet Normalize, RGB-вход, индексы всех пяти классов,
штатные аугментации и разбиение из `common.data_preparation`. Предсказание — CPU
`torch.long` с формой `[B,H,W]`, как у исходной модели.

Loss остаётся `WeightedCE + Dice`, коэффициенты 1 и 1. Веса классов вычисляются
только по train loader через `1/sqrt(freq)` и нормируются по среднему.
Optimizer — Adam, learning rate `1e-4`, обучение — 50 эпох. Бинарная смесь
CE/Dice/Focal/Jaccard из речной статьи здесь не используется.

`fit(train_loader, val_loader=None, save_best=True, patience=None, **kwargs)`
совместим с исходным вызовом. Если validation loader не передан, выполняются
50 эпох без выбора лучшей эпохи и early stopping. При его передаче лучшая
модель и состояние Adam восстанавливаются вместе. `load()` восстанавливает
также веса классов; он проверяет архитектуру и число классов. Файлы baseline
не подходят для загрузки в эту модель.

## Выбор модели

В существующем механизме выбора модели используются следующие импорт и класс:

```python
from UNETPP_EFFB3_RIVER.model import (
    UNetPlusPlusEffB3River,
    train_extra_transforms,
    val_extra_transforms,
)
ModelClass = UNetPlusPlusEffB3River
```

Готовый runner `run_pipeline_river.py` уже содержит этот выбор. Исходные
runner-файлы и подготовка данных сохранены.
Дополнительные библиотеки не нужны. Для `checkpoint(context_fn=...)` требуется
PyTorch 2.1 или новее; текущая среда использует PyTorch 2.7.1 и SMP 0.5.0.

Параметров 13 820 996 против 13 625 373 у baseline, прирост около 1,44%.
При ручном forward/backward/Adam с float32, batch=8 и входом 512×512 пик
выделенной CUDA-памяти составил 7 244 МиБ на RTX 3060. Это измерение одного
шага одной модели; полный pipeline также держит другие fold-модели на GPU.
Пересчёт активаций сохраняет batch size и точность float32.

Имя модели — `UN-WCD-RIVER`. Пути результатов:

- `UNETPP_EFFB3_RIVER/weights_{fold}.eth`
- `UNETPP_EFFB3_RIVER/raw_metrics.csv`
- `UNETPP_EFFB3_RIVER/interval_metrics.csv`

Обычный конструктор использует ImageNet weights. Для загрузки собственных
готовых весов без предварительного скачивания encoder можно создать
`UNetPlusPlusEffB3River(encoder_weights=None)`, затем вызвать `load(path)`.

## Оценка переноса

Сравнение с `UNETPP_EFFB3` должно использовать то же разбиение и текущие метрики,
особенно recall и Dice для вен и капилляров. Визуально следует оценивать тонкие
ветви, границы, пересечения и ложные слияния соседних сосудов. Attention не
гарантирует связность и не восстанавливает физиологический граф в пересечениях.
Результаты речных экспериментов нельзя считать результатами на сосудистых данных.
