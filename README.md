# Самостоятельная работа: восстановление сейсмических кубов

Срок сдачи — 29.11.2026 включительно. Основное описание задания находится в `slides.md`, пример оформления — в `TEMPLATE_independent_work_result.ipynb`.

Решение содержит отдельные ноутбуки для inpainting и outpainting. Обучение использует только соответствующий входной куб, без сторонних данных.

## Окружение и запуск

Для Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

Для Linux и macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Скачайте данные по [ссылке](https://disk.yandex.ru/d/JRT1Q5WP3NzGMQ) и разместите файлы так:

```text
input_seismic_data/
├── inpainting_cube_missed_data.npy
└── outpainting_cube_missed_data.npy
```

Откройте `inpainting.ipynb` или `outpainting.ipynb` и последовательно выполните все ячейки. Ноутбук проверит SHA-256, форму куба и расположение пропусков до начала обучения. Для продолжения обучения с лучшего чекпоинта той же конфигурации установите `RESUME = True` в блоке настройки обучения и повторно запустите ячейки обучения. После смены ориентации inpainting требуется полное переобучение: старые веса совместимы по форме, но обучены для другой оси и другой схемы маскировки.

## Текущая конфигурация

- Inpainting использует патчи `144×304` и целевую маску `103×213`. Куб транспонируется в `(y, z, x)` с формой `951×462×580`: на каждом y-срезе пропуск имеет размер `103×580`. Используются три позиции по `x`: `0:213`, `213:426`, `426:580`. Для каждой позиции выполняются 213 последовательных восстановлений `y=368:581`. Предыдущие y-срезы и левая восстановленная часть текущего среза используются как контекст. Outpainting использует патчи 368×256 и не изменён.
- В обоих ноутбуках `base_channels = 24`, в bottleneck добавлены свёртки с dilation 2 и 4.
- Inpainting: `BATCH_SIZE = 24`, `EPOCHS = 100`, `BASE_LR = 1e-3`, `L1_WEIGHT = 0.3`, `GRAD_LOSS_WEIGHT = 0.3`, `FFL_WEIGHT = 0.05`, `FFL_HF_CUTOFF = 0.25`, warmup 5 эпох, `RUN_NAME = "sequential-y-x-causal"`; `ReduceLROnPlateau(mode="max", factor=0.5, patience=4, min_lr=1e-5)` уменьшает LR по validation R.
- Размеры датасета inpainting: train — 5782 примера (826 y-центров × 7), validation — 600 (75 × 8), test — 152 (38 × 4). Validation и test используют held-out блок `z=76:179`, `x=0:580` (103×580 на y-срезе). Обучающие патчи берутся из `z=282:462` и не пересекают held-out или реальный пропуск. `TRAIN_SAMPLES_PER_CENTER` задаёт число случайных примеров на центральный y-срез за эпоху.
- Outpainting: `BATCH_SIZE = 16`, `EPOCHS = 75`, LR `2e-3`, `GRAD_LOSS_WEIGHT = 0.3`, `FFL_WEIGHT = 0.05`, `RUN_NAME = "gradient-l1-ffl"`; используется `ReduceLROnPlateau(patience=4, factor=0.5)`.
- Потеря включает L1, MSE, gradient L1 и `0.5 · (1 − R)`. В inpainting и outpainting используется FFL по высоким частотам: спектр `rfft2((pred − target) · mask, norm="ortho")`, cutoff `0.25`, вес `0.05` и линейный warmup 5 эпох. На test применяется полный вес. Норма градиентов не вычисляется.
- Положение патчей и масштаб амплитуд остаются динамическими. В inpainting отражение выполняется только по `z`, чтобы не менять направление последовательного восстановления по `x`. Gaussian noise предрассчитывается перед обучением в банк из 64 вариантов. Нормализация использует только train-срезы.
- Все блоки внимания заменены на универсальный `TokenAttention`: токенизация через `Conv2d`, размерность токена 128, 4 головы. На 1/16 полное внимание использует 171 токен; на 1/8, 1/4 и 1/2 — 171, 684 и 2736 токенов. В decoder query создаются только для центральных 20% пропуска (3, 15 и 55 query), которые обращаются ко всем key/value своего масштаба. Крайние токены не участвуют в attention. Обратное преобразование выполняется через `ConvTranspose2d`, блок добавляет результат к входу через residual.
- Для ablation без gradient loss установите `GRAD_LOSS_WEIGHT = 0.0` и `RUN_NAME = "no-grad-l1"`, сохранив остальные параметры.

Выходы ячеек очищены, поскольку прежние результаты относились к другим конфигурациям.

## Структура проекта

```text
├── 0_seismic.ipynb
├── inpainting.ipynb
├── outpainting.ipynb
├── seismic_dataset.py
├── inference/
├── SeismicUtils.py
├── TEMPLATE_independent_work_result.ipynb
├── requirements.txt
├── slides.md
└── README.md
```

## Результаты обучения

| Куб | R train | R val | R test |
| --- | ---: | ---: | ---: |
| Inpainting | требуется запуск | требуется запуск | требуется запуск |
| Outpainting | требуется запуск | требуется запуск | требуется запуск |

Актуальная конфигурация ещё не обучалась. В репозитории нет входных данных и обученных весов. После выполнения ноутбуков итоговые значения R для train, val и test выводятся в блоке тестирования; их необходимо перенести в таблицу.

После обучения сохраняются следующие артефакты:

- `inpainting_unet_inference.pt` и `inpainting_cube_restored.npy`;
- `outpainting_unet_inference.pt` и `outpainting_cube_restored.npy`;
- `inpainting_training_history.png` и `outpainting_training_history.png`;
- лучшие чекпоинты `best_inpainting_unet.pt` и `best_outpainting_unet.pt`;
- журналы TensorBoard в `runs/`.

Крупные данные, веса и журналы исключены из Git через `.gitignore`.
