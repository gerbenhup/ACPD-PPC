import numpy as np
import tensorflow as tf

def uniform(Nf: int) -> tf.Tensor:
    """Uniform weights of the forecast losses of the Nf future steps, used by the forecast losses

    Args:
        Nf (int): number of future steps

    Returns:
        tf.Tensor: weights, shape [Nf], each equal to 1/Nf
    """

    omega = np.ones(Nf, dtype=np.float32) / Nf
    omega = tf.constant(omega)

    return omega
