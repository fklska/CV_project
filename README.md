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

- Inpainting использует патчи 368×176: этот размер помещается во все четыре известные области. Для верхней области вертикальная позиция фиксирована, для нижней доступны три позиции. Outpainting использует патчи 368×256.
- Обучающие патчи, маски и аугментации генерируются заново при каждом обращении к примеру. Validation и test остаются фиксированными.
- `BATCH_SIZE = 16`, в bottleneck добавлены свёртки с dilation 2 и 4.
- Потеря включает L1, MSE, исправленный gradient L1, Sobel/Laplacian L1 на масштабах 1, 1/2 и 1/4, а также `0.5 · (1 − R)`.
- Основной запуск: `GRAD_LOSS_WEIGHT = 0.3`, `HIGH_FREQ_LOSS_WEIGHT = 0.1`, `RUN_NAME = "grad-l1"`.
- Для ablation без перворазностного gradient loss установите `GRAD_LOSS_WEIGHT = 0.0` и `RUN_NAME = "no-grad-l1"`, сохранив остальные параметры, и заново выполните все ячейки. Запуски сравниваются по лучшему validation R.

Старые выходы ячеек очищены, поскольку они относились к прежней конфигурации.

## Структура проекта

```text
├── 0_seismic.ipynb
├── inpainting.ipynb
├── outpainting.ipynb
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
