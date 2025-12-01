"""
data_utils.py
-------------

Функції для:
- завантаження даних з CSV
- розбиття на train/test
- (опційної) нормалізації x, t
- підготовки numpy-масивів для моделі
"""

from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import config


def load_dataset(path: str) -> pd.DataFrame:
    """
    Завантажити датасет з CSV-файлу.

    Parameters
    ----------
    path : str
        Шлях до файлу (наприклад, "data/data_sampled.csv").

    Returns
    -------
    df : pandas.DataFrame
        Таблиця з даними.
    """
    df = pd.read_csv(path)
    return df


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбити дані на train/test шляхом випадкового перемішування.

    Parameters
    ----------
    df : pandas.DataFrame
        Повний датасет.
    test_size : float
        Частка тестових даних.
    random_state : int
        Для відтворюваності.

    Returns
    -------
    df_train, df_test : pandas.DataFrame
        Тренувальна та тестова частина.
    """
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_test = int(n_total * test_size)

    df_test = df_shuffled.iloc[:n_test].reset_index(drop=True)
    df_train = df_shuffled.iloc[n_test:].reset_index(drop=True)
    return df_train, df_test


def normalize_columns(
    x: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Масштабує x і t в [0, 1].

    Returns також словник з параметрами нормалізації.
    """
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()

    x_norm = (x - x_min) / (x_max - x_min)
    t_norm = (t - t_min) / (t_max - t_min)

    scalers = {
        "x_min": x_min,
        "x_max": x_max,
        "t_min": t_min,
        "t_max": t_max,
    }
    return x_norm, t_norm, scalers


def apply_normalization(
    x: np.ndarray,
    t: np.ndarray,
    scalers: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Застосувати збережені параметри нормалізації до нових даних.
    """
    x_min, x_max = scalers["x_min"], scalers["x_max"]
    t_min, t_max = scalers["t_min"], scalers["t_max"]

    x_norm = (x - x_min) / (x_max - x_min)
    t_norm = (t - t_min) / (t_max - t_min)
    return x_norm, t_norm


def prepare_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Підготовка numpy-масивів для train/test.

    Очікується, що у DataFrame є колонки:
    - config.X_COLUMN (x)
    - config.TIME_COLUMN (t)
    - config.FIELD_COLUMN (T, тобто u)

    Returns
    -------
    train_data, test_data : dict
        Словники з ключами 'x', 't', 'u'.
    scalers : dict
        Параметри нормалізації (якщо увімкнена).
    """
    x_train = df_train[config.X_COLUMN].values.reshape(-1, 1)
    t_train = df_train[config.TIME_COLUMN].values.reshape(-1, 1)
    u_train = df_train[config.FIELD_COLUMN].values.reshape(-1, 1)

    x_test = df_test[config.X_COLUMN].values.reshape(-1, 1)
    t_test = df_test[config.TIME_COLUMN].values.reshape(-1, 1)
    u_test = df_test[config.FIELD_COLUMN].values.reshape(-1, 1)

    scalers: Dict[str, Any] = {}

    if config.NORMALIZE_INPUTS:
        x_train, t_train, scalers = normalize_columns(x_train, t_train)
        x_test, t_test = apply_normalization(x_test, t_test, scalers)

    train_data = {"x": x_train, "t": t_train, "u": u_train}
    test_data = {"x": x_test, "t": t_test, "u": u_test}
    return train_data, test_data, scalers
