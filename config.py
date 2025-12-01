"""
config.py
---------

Усі налаштування проєкту в одному місці:
- шляхи до файлів з даними
- назви колонок
- гіперпараметри навчання
- директорія для збереження графіків
"""

# Шляхи до файлів з даними
FULL_DATA_PATH = "data/data.csv"
SAMPLED_DATA_PATH = "data/data_sampled.csv"

# Назви колонок у CSV
X_COLUMN = "x"      # координата простору
TIME_COLUMN = "t"   # час
FIELD_COLUMN = "T"  # фізична величина u(x, t), температура

# Нормалізація x і t в [0, 1]
NORMALIZE_INPUTS = True

# Коефіцієнт теплопровідності в PDE:
# du/dt = alpha * d^2 u / dx^2
ALPHA = 0.01

# Гіперпараметри навчання
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# Як часто друкувати логи
LOG_EVERY = 100

# Директорія для збереження графіків
IMG_DIR = "data/img"
