"""
trainer.py
----------

Функції для навчання та оцінки PINN.
"""

from typing import Dict, List

import numpy as np
import tensorflow as tf

import config
from physics import physics_loss


def numpy_to_tensor(data: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    """
    Перетворити numpy-масиви 'x', 't', 'u' у tf.Tensor.
    """
    return {
        "x": tf.convert_to_tensor(data["x"], dtype=tf.float32),
        "t": tf.convert_to_tensor(data["t"], dtype=tf.float32),
        "u": tf.convert_to_tensor(data["u"], dtype=tf.float32),
    }


def train_pinn(
    model: tf.keras.Model,
    train_data_np: Dict[str, np.ndarray],
) -> Dict[str, List[float]]:
    """
    Навчання PINN'а.

    Лосс:
        total_loss = data_loss + physics_loss
    """
    optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)

    train_data = numpy_to_tensor(train_data_np)
    x_tf, t_tf, u_tf = train_data["x"], train_data["t"], train_data["u"]

    inputs_tf = tf.concat([x_tf, t_tf], axis=1)

    history = {
        "total_loss": [],
        "data_loss": [],
        "physics_loss": [],
    }

    for epoch in range(config.NUM_EPOCHS):
        with tf.GradientTape() as tape:
            phys_loss_value = physics_loss(model, x_tf, t_tf, alpha=config.ALPHA)
            u_pred = model(inputs_tf)
            data_loss_value = tf.reduce_mean(tf.square(u_pred - u_tf))
            total_loss_value = phys_loss_value + data_loss_value

        grads = tape.gradient(total_loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        history["total_loss"].append(float(total_loss_value.numpy()))
        history["data_loss"].append(float(data_loss_value.numpy()))
        history["physics_loss"].append(float(phys_loss_value.numpy()))

        if epoch % config.LOG_EVERY == 0:
            print(
                f"Epoch {epoch:5d}/{config.NUM_EPOCHS} | "
                f"Total: {total_loss_value.numpy():.6e} | "
                f"Data: {data_loss_value.numpy():.6e} | "
                f"Physics: {phys_loss_value.numpy():.6e}"
            )

    return history


def evaluate_pinn(
    model: tf.keras.Model,
    test_data_np: Dict[str, np.ndarray],
) -> float:
    """
    Оцінити MSE на тестовій вибірці.
    """
    test_data = numpy_to_tensor(test_data_np)
    x_tf, t_tf, u_true = test_data["x"], test_data["t"], test_data["u"]

    inputs_tf = tf.concat([x_tf, t_tf], axis=1)
    u_pred = model(inputs_tf)

    mse_tensor = tf.reduce_mean(tf.square(u_pred - u_true))
    mse = float(mse_tensor.numpy())
    return mse
