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

Откройте `inpainting.ipynb` или `outpainting.ipynb` и последовательно выполните все ячейки. Ноутбук проверит SHA-256, форму куба и расположение пропусков до начала обучения. Для продолжения обучения с лучшего чекпоинта той же архитектуры установите `RESUME = True` в блоке настройки обучения и повторно запустите ячейки обучения. Чекпоинты прежней архитектуры с новой моделью несовместимы.

## Текущая конфигурация

- Inpainting использует патчи 304×144 с дырой 213×103: 21 837 известных и 21 939 восстанавливаемых пикселей. На инференсе модель получает центрированный crop `y=323:627`, `z=159:303`, а результат записывается только в реальную дыру. Outpainting использует патчи 368×256.
- В обоих ноутбуках `base_channels = 24`, в bottleneck добавлены свёртки с dilation 2 и 4.
- Inpainting: `BATCH_SIZE = 24`, `EPOCHS = 100`, `BASE_LR = 2e-3`, `GRAD_LOSS_WEIGHT = 0.75`, `FFL_WEIGHT = 0.05`, `RUN_NAME = "block-attn-ffl"`; LR имеет 3 эпохи warmup и cosine-затухание до `1e-5`.
- Размеры датасета inpainting: train — 3000 примеров (500 центров × 6), validation — 360 (45 × 8), test — 92 (23 × 4). Validation и test строятся на held-out блоке `y=77:290`, `z=179:282` (213×103 во всех срезах), который не попадает в train ни как target, ни как input (обучающие патчи, пересекающие блок, отбрасываются). `TRAIN_SAMPLES_PER_CENTER` задаёт число случайных пространственных примеров на центральный срез за эпоху.
- Outpainting: `BATCH_SIZE = 16`, `EPOCHS = 75`, LR `2e-3`, `GRAD_LOSS_WEIGHT = 0.3`, `FFL_WEIGHT = 0.05`, `RUN_NAME = "gradient-l1-ffl"`; используется `ReduceLROnPlateau(patience=4, factor=0.5)`.
- Потеря включает L1, MSE, gradient L1, focal frequency loss (FFL) по высоким частотам (`rfft2`, низкие частоты ниже `FFL_HF_CUTOFF = 0.25` отключены; вес `0.05` растёт за 5 эпох, на test — полный) и `0.5 · (1 − R)`. Норма градиентов не вычисляется.
- Положение патчей, отражения и масштаб амплитуд остаются динамическими. Gaussian noise предрассчитывается перед обучением в банк из 64 вариантов. Нормализация использует только train-срезы.
- RingAttention обрабатывает весь батч одновременно. Query объединяются в блоки 2×2; фиксированные радиальные зоны: центральные 10% дыры видят все попиксельные key/value окна 115×60 (6900 позиций), остальные query — только внешнее кольцо шириной 4 (1336 позиций). Вес выхода attention кусочно-постоянный по зонам (1.0 / 0.75 / 0.25), 35 внешних пикселей дыры исключены из query. К логитам добавлен обучаемый относительный bias по (dy, dx); дополнительно глобальный self-attention на 1/16 и 1/8.
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
