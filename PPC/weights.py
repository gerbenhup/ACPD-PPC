import numpy as np
import tensorflow as tf

def Uniform(Nf: int) -> tf.Tensor:
    """Generate uniform time weights for MaximumLogLikelihoodLoss

    Args:
        Nf (int): number of future predictions

    Returns:
        tf.Tensor: uniform weights, each equal to 1/Nf
    """

    omega = np.ones(Nf, dtype=np.float32) / Nf
    omega = tf.constant(omega)

    return omega
