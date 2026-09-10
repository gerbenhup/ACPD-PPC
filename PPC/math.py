import tensorflow as tf

def log(x: tf.Tensor) -> tf.Tensor:
    """Safe logarithm function. Inputs are clipped to avoid non-allowed input values.
    Clipped at (almost) zero. Gradients not passing whenever input is not safe.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe logarithm of input value
    """

    x_clipped = tf.math.maximum(x, 1.1754943e-38)
    x_log     = tf.math.log(x_clipped)

    return x_log

def exp(x: tf.Tensor) -> tf.Tensor:
    """Safe exponent function. Inputs are clipped to avoid non-allowed input values.
    Clipped at log(tf.float32.max). Gradients not passing whenever input is not safe.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe exponent of input value
    """

    x_clipped = tf.math.minimum(x, 88.7228)
    x_exp     = tf.math.exp(x_clipped)

    return x_exp

def square(x: tf.Tensor) -> tf.Tensor:
    """Safe square function. Inputs are clipped to avoid non-allowed input values.
    Clipped at sqrt(tf.float32.max). Gradients not passing whenever input is not safe.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe square of input value
    """

    x_abs     = tf.math.abs(x)
    x_clipped = tf.math.minimum(x_abs, 1.8446743e+19)
    x_square  = tf.math.square(x_clipped)

    return x_square
