import tensorflow as tf

from . import probability

class MeanAbsoluteError:
    """Mean absolute error reconstruction loss"""

    def __init__(self):
        pass

    def __call__(self, samples_past: tf.Tensor, decodings_past: tf.Tensor, samples_future: tf.Tensor, decodings_future: tf.Tensor) -> tf.Tensor:
        """Mean absolute error between the past and future samples and their reconstructions

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            decodings_past (tf.Tensor): reconstructed past samples, shape [B, Np, ...]
            samples_future (tf.Tensor): future samples, shape [B, Nf, ...]
            decodings_future (tf.Tensor): reconstructed future samples, shape [B, Nf, ...]

        Returns:
            tf.Tensor: loss, scalar
        """

        samples   = tf.concat([samples_past, samples_future], axis=1)
        decodings = tf.concat([decodings_past, decodings_future], axis=1)

        return tf.math.reduce_mean(tf.math.abs(samples - decodings))

class MeanSquaredError:
    """Mean squared error reconstruction loss"""

    def __init__(self):
        pass

    def __call__(self, samples_past: tf.Tensor, decodings_past: tf.Tensor, samples_future: tf.Tensor, decodings_future: tf.Tensor) -> tf.Tensor:
        """Mean squared error between the past and future samples and their reconstructions

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            decodings_past (tf.Tensor): reconstructed past samples, shape [B, Np, ...]
            samples_future (tf.Tensor): future samples, shape [B, Nf, ...]
            decodings_future (tf.Tensor): reconstructed future samples, shape [B, Nf, ...]

        Returns:
            tf.Tensor: loss, scalar
        """

        samples   = tf.concat([samples_past, samples_future], axis=1)
        decodings = tf.concat([decodings_past, decodings_future], axis=1)

        return tf.math.reduce_mean(tf.math.squared_difference(samples, decodings))

class MaximumLogLikelihoodLoss:
    """Negative log-likelihood forecast loss, under a Gaussian or Laplace prior on the encodings"""

    def __init__(self, omega: tf.Tensor, warmup: bool, distribution: str='Gaussian'):
        """
        Args:
            omega (tf.Tensor): weights of the losses of the Nf future steps, shape [Nf], e.g. weights.uniform(Nf)
            warmup (bool): whether to ignore the predicted standard deviations/scales and assume unit ones (True), as used during warm-up, or not (False)
            distribution (str, optional): distribution of the forecasted encodings, 'Gaussian' or 'Laplace'. Defaults to 'Gaussian'.
        """

        self.omega = omega

        if distribution == 'Gaussian':
            if warmup:
                self.loss_func = probability.gaussian_log_likelihood_unit_variance
            else:
                self.loss_func = probability.gaussian_log_likelihood

        elif distribution == 'Laplace':
            if warmup:
                self.loss_func = probability.laplace_log_likelihood_unit_scale
            else:
                self.loss_func = probability.laplace_log_likelihood

        else:
            raise ValueError('Distribution {} unknown'.format(distribution))

    def __call__(self, encodings_future_true: tf.Tensor, encodings_future_pred: tf.Tensor) -> tf.Tensor:
        """Negative log-likelihood of the true future encodings, averaged over the batch and weighted over the future steps

        Args:
            encodings_future_true (tf.Tensor): encodings of the true future samples, shape [B, Nf, Ne]
            encodings_future_pred (tf.Tensor): predicted distributions of the future encodings, shape [B, Nf, Ne, 2]

        Returns:
            tf.Tensor: loss, scalar
        """

        losses        = -self.loss_func(encodings_future_true, encodings_future_pred) # [B, Nf]
        batch_loss    = tf.math.reduce_mean(losses, axis=0) # [Nf,]
        weighted_loss = tf.tensordot(batch_loss, self.omega, 1) # []

        return weighted_loss

class ContrastiveNLLRLoss:
    """Contrastive negative log-likelihood ratio forecast loss, used by the PCPC trainer

    The loss is log(mean likelihood of the negative samples) - log(likelihood of the positive sample),
    averaged over the batch. The mean over the negative samples is computed in log space with
    log-sum-exp, so it neither underflows nor overflows for high-dimensional encodings.
    """

    def __init__(self, omega: tf.Tensor, warmup: bool):
        """
        Args:
            omega (tf.Tensor): weights of the losses of the Nf future steps, shape [Nf], e.g. weights.uniform(Nf)
            warmup (bool): whether to ignore the predicted standard deviations and assume unit ones (True), as used during warm-up, or not (False)
        """

        self.omega  = omega
        self.warmup = warmup

        if warmup:
            self.loss_func = probability.gaussian_log_likelihood_unit_variance
        else:
            self.loss_func = probability.gaussian_log_likelihood

    def __call__(self, encodings_future_pos: tf.Tensor, encodings_future_neg: tf.Tensor, encodings_future_pred: tf.Tensor) -> tf.Tensor:
        """Contrastive loss of the positive future encodings against the negative ones

        Args:
            encodings_future_pos (tf.Tensor): encodings of the true (positive) future samples, shape [B, Nf, Ne]
            encodings_future_neg (tf.Tensor): encodings of the negative future samples, shape [B, Nchi, Nf, Ne]
            encodings_future_pred (tf.Tensor): predicted Gaussian distributions of the future encodings, shape [B, Nf, Ne, 2]

        Returns:
            tf.Tensor: loss, scalar
        """

        # Determine log-likelihoods of the negative samples
        encodings_future_neg = tf.transpose(encodings_future_neg, (1, 0, 2, 3)) # [Nchi, B, Nf, Ne]
        log_neg              = tf.map_fn(lambda x: self.loss_func(x, encodings_future_pred), encodings_future_neg) # [Nchi, B, Nf]

        # Determine logarithm of the mean likelihood of the negative samples: log(mean(exp(log_neg))) = logsumexp(log_neg) - log(Nchi)
        Nchi         = tf.cast(tf.shape(log_neg)[0], log_neg.dtype)
        log_mean_neg = tf.math.reduce_logsumexp(log_neg, axis=0) - tf.math.log(Nchi) # [B, Nf]

        # Determine log-likelihood of the positive samples
        log_pos = self.loss_func(encodings_future_pos, encodings_future_pred) # [B, Nf]

        # Reduce over batch dimension and take weighted average
        losses        = tf.math.reduce_mean(log_mean_neg - log_pos, axis=0) # [Nf,]
        weighted_loss = tf.tensordot(losses, self.omega, 1) # []

        return weighted_loss
