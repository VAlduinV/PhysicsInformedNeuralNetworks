"""
pinn_model.py
-------------

Архітектура Physics-Informed Neural Network (PINN).
"""

import tensorflow as tf


class PINN(tf.keras.Model):
    """
    Проста повнозв'язна нейромережа для апроксимації u(x, t).

    Вхід: 2 нейрони (x, t)
    Вихід: 1 нейрон (u)
    """

    def __init__(self, hidden_units: int = 50, num_hidden_layers: int = 2):
        """
        Parameters
        ----------
        hidden_units : int
            Кількість нейронів у прихованих шарах.
        num_hidden_layers : int
            Кількість прихованих шарів.
        """
        super().__init__()

        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_units, activation="tanh")
            for _ in range(num_hidden_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Прямий прохід.

        Parameters
        ----------
        inputs : tf.Tensor, shape (batch_size, 2)
            inputs[:, 0] = x, inputs[:, 1] = t

        Returns
        -------
        u_pred : tf.Tensor, shape (batch_size, 1)
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        u_pred = self.output_layer(x)
        return u_pred
