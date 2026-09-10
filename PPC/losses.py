import tensorflow as tf

from . import probability

class MeanSquaredError:
    """Mean squared error reconstruction loss"""

    def __init__(self):
        pass

    def __call__(self, samples_past: tf.Tensor, decodings_past: tf.Tensor, samples_future: tf.Tensor, decodings_future: tf.Tensor) -> tf.Tensor:
        """Call function

        Args:
            samples_past (tf.Tensor): Samples in past, [B, Np, ...]
            decodings_past (tf.Tensor): Decoded encodings in past, [B, Np, ...]
            samples_future (tf.Tensor): Samples in future, [B, Nf, ...]
            decodings_future (tf.Tensor): Decoded encodings in future, [B, Nf, ...]

        Returns:
            tf.Tensor: loss values
        """

        samples   = tf.concat([samples_past, samples_future], axis=1)
        decodings = tf.concat([decodings_past, decodings_future], axis=1)

        return tf.math.reduce_mean(tf.math.squared_difference(samples, decodings))

class MaximumLogLikelihoodLoss:

    def __init__(self, omega: tf.Tensor, warmup: bool):
        """Init function

        Args:
            omega (tf.Tensor): weights of losses for each time step
            warmup (bool): whether to assume unit standard deviation (True) or not (False)
        """

        self.omega = omega

        if warmup:
            self.loss_func = probability.GaussianLogLikelihoodUnitVariance
        else:
            self.loss_func = probability.GaussianLogLikelihood

    def __call__(self, encodings_future_true: tf.Tensor, encodings_future_pred: tf.Tensor) -> tf.Tensor:
        """Call function

        Args:
            encodings_future_true (tf.Tensor): encodings of future true samples (shape [B, Nf, Ne])
            encodings_future_pred (tf.Tensor): encodings of future predicted samples (shape [B, Nf, Ne, 2])

        Returns:
            tf.Tensor: loss values
        """

        losses        = -self.loss_func(encodings_future_true, encodings_future_pred) # [B, Nf]
        batch_loss    = tf.math.reduce_mean(losses, axis=0) # [Nf,]
        weighted_loss = tf.tensordot(batch_loss, self.omega, 1) # []

        return weighted_loss
