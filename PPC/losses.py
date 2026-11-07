import tensorflow as tf

from . import probability, math

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
    """Negative log-likelihood forecast loss, under a Gaussian or Laplace prior on the encodings"""

    def __init__(self, omega: tf.Tensor, warmup: bool, distribution: str='Gaussian'):
        """Init function

        Args:
            omega (tf.Tensor): weights of losses for each time step
            warmup (bool): whether to assume unit standard deviation/scale (True) or not (False)
            distribution (str, optional): prior distribution of the forecasted encodings, 'Gaussian' or 'Laplace'. Defaults to 'Gaussian'.
        """

        self.omega = omega

        if distribution == 'Gaussian':
            if warmup:
                self.loss_func = probability.GaussianLogLikelihoodUnitVariance
            else:
                self.loss_func = probability.GaussianLogLikelihood

        elif distribution == 'Laplace':
            if warmup:
                self.loss_func = probability.LaplaceLogLikelihoodUnitScale
            else:
                self.loss_func = probability.LaplaceLogLikelihood

        else:
            raise ValueError('Distribution {} unknown'.format(distribution))

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

class ContrastiveNLLRLoss:
    """Contrastive negative log-likelihood ratio forecast loss, used by the PCPC trainer"""

    def __init__(self, omega: tf.Tensor, warmup: bool):
        """Init function

        Args:
            omega (tf.Tensor): weights of losses for each time step
            warmup (bool): whether to assume unit standard deviation (True) or not (False)
        """

        self.omega  = omega
        self.warmup = warmup

        if warmup:
            self.loss_func    = probability.GaussianLikelihoodUnitVariance
            self.logloss_func = probability.GaussianLogLikelihoodUnitVariance
        else:
            self.loss_func    = probability.GaussianLikelihood
            self.logloss_func = probability.GaussianLogLikelihood

    def __call__(self, encodings_future_pos: tf.Tensor, encodings_future_neg: tf.Tensor, encodings_future_pred: tf.Tensor) -> tf.Tensor:
        """Call function

        Args:
            encodings_future_pos (tf.Tensor): encodings of future positive samples (shape [B, Nf, Ne])
            encodings_future_neg (tf.Tensor): encodings of future negative samples (shape [B, Nchi, Nf, Ne])
            encodings_future_pred (tf.Tensor): encodings of future forecasted samples (shape [B, Nf, Ne, 2])

        Returns:
            tf.Tensor: contrastive NLLR loss of this batch
        """

        # Determine logarithm of the mean likelihood for the negative samples
        encodings_future_neg = tf.transpose(encodings_future_neg, (1, 0, 2, 3)) # [Nchi, B, Nf, Ne]
        mean_neg             = tf.math.reduce_mean(tf.map_fn(lambda x: self.loss_func(x, encodings_future_pred), encodings_future_neg), axis=0) # [B, Nf]
        log_mean_neg         = math.log(mean_neg) # [B, Nf]

        # Determine logarithm of the positive sample likelihood
        log_pos = self.logloss_func(encodings_future_pos, encodings_future_pred) # [B, Nf]

        # Reduce over batch dimension and take weighted average
        losses        = tf.math.reduce_mean(log_mean_neg - log_pos, axis=0) # [Nf,]
        weighted_loss = tf.tensordot(losses, self.omega, 1) # []

        return weighted_loss
