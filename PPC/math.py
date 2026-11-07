import tensorflow as tf

def log(x: tf.Tensor) -> tf.Tensor:
    """Logarithm that is safe for inputs <= 0: inputs are clipped to the smallest positive normal float32.
    No gradient flows through clipped inputs.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe logarithm of input value
    """

    x_clipped = tf.math.maximum(x, 1.1754943e-38)
    x_log     = tf.math.log(x_clipped)

    return x_log

def exp(x: tf.Tensor) -> tf.Tensor:
    """Exponent that does not overflow: inputs are clipped at log(tf.float32.max).
    No gradient flows through clipped inputs.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe exponent of input value
    """

    x_clipped = tf.math.minimum(x, 88.7228)
    x_exp     = tf.math.exp(x_clipped)

    return x_exp

def square(x: tf.Tensor) -> tf.Tensor:
    """Square that does not overflow: absolute inputs are clipped at sqrt(tf.float32.max).
    No gradient flows through clipped inputs.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: safe square of input value
    """

    x_abs     = tf.math.abs(x)
    x_clipped = tf.math.minimum(x_abs, 1.8446743e+19)
    x_square  = tf.math.square(x_clipped)

    return x_square

def sqrt(x: tf.Tensor) -> tf.Tensor:
    """Square root; no clipping, as no operation is expected to give unsafe inputs. Kept for symmetry with the other functions.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: square root of input value
    """

    return tf.math.sqrt(x)

def abs(x: tf.Tensor) -> tf.Tensor:
    """Absolute value; no clipping, as no operation is expected to give unsafe inputs. Kept for symmetry with the other functions.

    Args:
        x (tf.Tensor): input value

    Returns:
        tf.Tensor: absolute value of input value
    """

    return tf.math.abs(x)
