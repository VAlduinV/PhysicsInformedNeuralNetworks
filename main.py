"""
main.py
-------

Точка входу проєкту Physics-Informed Neural Networks.

Для зручності:
- спочатку тренуємося на data_sampled.csv
- потім на повному data.csv

Для кожного датасету:
- завантажуємо, ділимо на train/test, нормалізуємо
- навчаємо PINN
- рахуємо тестовий MSE
- зберігаємо два графіки в data/img:
    <prefix>_loss.png
    <prefix>_prediction.png
"""

import numpy as np
import tensorflow as tf  # тільки для "прогріву" моделі

import config
from data_utils import load_dataset, train_test_split, prepare_data
from pinn_model import PINN
from trainer import train_pinn, evaluate_pinn
from visualization import save_loss_history, save_prediction_vs_true


def run_for_dataset(dataset_path: str, prefix: str) -> None:
    """
    Запустити повний пайплайн (train + eval + графіки) для одного датасету.

    Parameters
    ----------
    dataset_path : str
        Шлях до CSV-файлу.
    prefix : str
        Префікс (імі'я) цього запуску для логів і картинок.
    """
    print("=" * 80)
    print(f"DATASET: {prefix} | path = {dataset_path}")
    print("=" * 80)

    # 1. Завантаження
    print("Loading dataset...")
    df = load_dataset(dataset_path)

    # 2. Train/test split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=config.RANDOM_STATE
    )
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # 3. Підготовка даних
    train_data, test_data, scalers = prepare_data(df_train, df_test)
    if config.NORMALIZE_INPUTS:
        print("Inputs (x, t) were normalized to [0, 1].")

    # 4. Створення моделі
    model = PINN(hidden_units=50, num_hidden_layers=2)

    # Невеликий "прогрів", щоб побудувати граф:
    dummy_input = np.concatenate(
        [train_data["x"][:5], train_data["t"][:5]], axis=1
    ).astype("float32")
    _ = model(tf.convert_to_tensor(dummy_input))

    # 5. Навчання
    print("Starting training...")
    history = train_pinn(model, train_data)
    print("Training finished.")

    # 6. Оцінка
    mse_test = evaluate_pinn(model, test_data)
    print(f"[{prefix}] Test MSE: {mse_test:.6e}")

    # 7. Збереження графіків
    loss_path = save_loss_history(history, prefix=prefix)
    print(f"[{prefix}] Loss plot saved to: {loss_path}")

    x_test = test_data["x"]
    t_test = test_data["t"]
    u_true = test_data["u"]

    inputs_test = np.concatenate([x_test, t_test], axis=1).astype("float32")
    u_pred = model(inputs_test).numpy()

    pred_path = save_prediction_vs_true(
        x_test, t_test, u_true, u_pred, prefix=prefix
    )
    print(f"[{prefix}] Prediction plot saved to: {pred_path}")


def main() -> None:
    """
    Основна функція:
    - запускає пайплайн для data_sampled.csv
    - потім для data.csv
    """
    # Спочатку — менший датасет
    run_for_dataset(config.SAMPLED_DATA_PATH, prefix="sampled")

    # Потім — повний датасет
    run_for_dataset(config.FULL_DATA_PATH, prefix="full")


if __name__ == "__main__":
    main()
