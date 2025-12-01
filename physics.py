"""
physics.py
----------

Фізичний лосс для рівняння теплопровідності:

    du/dt = alpha * d²u/dx²
"""

import tensorflow as tf

import config


def pde_residual(
    model: tf.keras.Model,
    x: tf.Tensor,
    t: tf.Tensor,
    alpha: float,
) -> tf.Tensor:
    """
    Обчислити залишок PDE:
        r = du/dt - alpha * d²u/dx²
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)

        inputs = tf.concat([x, t], axis=1)
        u_pred = model(inputs)

        # Перші похідні
        u_x = tape.gradient(u_pred, x)
        u_t = tape.gradient(u_pred, t)

    # Друга похідна по x
    u_xx = tape.gradient(u_x, x)

    del tape

    residual = u_t - alpha * u_xx
    return residual


def physics_loss(
    model: tf.keras.Model,
    x: tf.Tensor,
    t: tf.Tensor,
    alpha: float = config.ALPHA,
) -> tf.Tensor:
    """
    Фізичний лосс = MSE(residual).
    """
    residual = pde_residual(model, x, t, alpha)
    loss = tf.reduce_mean(tf.square(residual))
    return loss
