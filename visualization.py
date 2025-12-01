"""
visualization.py
----------------

Збереження графіків:
- історія лоссів
- порівняння u_true та u_pred

Усі картинки зберігаються в config.IMG_DIR.
"""

from typing import Dict, List

import os
import numpy as np
import matplotlib.pyplot as plt

import config


def _ensure_img_dir_exists() -> None:
    """
    Створити директорію для картинок, якщо її ще немає.
    """
    os.makedirs(config.IMG_DIR, exist_ok=True)


def save_loss_history(
    history: Dict[str, List[float]],
    prefix: str,
) -> str:
    """
    Зберегти графік історії лоссів у файл.

    Parameters
    ----------
    history : dict
        Дані від train_pinn().
    prefix : str
        Префікс для імені файлу (наприклад, "sampled" або "full").

    Returns
    -------
    path : str
        Повний шлях до збереженого файлу.
    """
    _ensure_img_dir_exists()

    epochs = np.arange(len(history["total_loss"]))

    plt.figure()
    plt.plot(epochs, history["total_loss"], label="Total loss")
    plt.plot(epochs, history["data_loss"], label="Data loss")
    plt.plot(epochs, history["physics_loss"], label="Physics loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"PINN training loss ({prefix})")
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(config.IMG_DIR, f"{prefix}_loss.png")
    plt.savefig(path)
    plt.close()
    return path


def save_prediction_vs_true(
    x: np.ndarray,
    t: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    prefix: str,
    num_points: int = 200,
) -> str:
    """
    Зберегти графік True vs Pred (по підмножині точок).

    Для простоти беремо перші num_points точок,
    сортуючи їх за x.
    """
    _ensure_img_dir_exists()

    n = x.shape[0]
    num_points = min(num_points, n)

    idx = np.argsort(x.flatten())[:num_points]
    x_plot = x[idx].flatten()
    u_true_plot = u_true[idx].flatten()
    u_pred_plot = u_pred[idx].flatten()

    plt.figure()
    plt.plot(x_plot, u_true_plot, "o", label="True T(x, t)", alpha=0.7)
    plt.plot(x_plot, u_pred_plot, "-", label="PINN T(x, t)")
    plt.xlabel("x (normalized)" if config.NORMALIZE_INPUTS else "x")
    plt.ylabel("T")
    plt.title(f"True vs PINN prediction ({prefix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(config.IMG_DIR, f"{prefix}_prediction.png")
    plt.savefig(path)
    plt.close()
    return path
