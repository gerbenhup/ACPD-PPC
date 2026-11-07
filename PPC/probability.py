"""Likelihoods, distances and tail probabilities of encodings under the predicted distributions.

Predicted distributions have shape [..., Ne, 2]: the means (locations) and standard deviations (scales)
of independent distributions per encoding dimension.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Lower bound on the standard deviations (scales), for numerical stability
MIN_SCALE = 1e-9

def gaussian_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood of the true encodings, computed as the exponent of the log-likelihood

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood, shape [...]
    """

    return tf.math.exp(gaussian_log_likelihood(encodings_true, encodings_pred))

def gaussian_likelihood_unit_variance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood of the true encodings with unit variance, computed as the exponent of the log-likelihood

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]; only the means are used

    Returns:
        tf.Tensor: likelihood, shape [...]
    """

    return tf.math.exp(gaussian_log_likelihood_unit_variance(encodings_true, encodings_pred))

def gaussian_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log-likelihood of the true encodings, summed over the encoding dimensions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    stds = tf.math.maximum(encodings_pred[..., 1], MIN_SCALE)

    return -encodings_pred.shape[-2]*tf.math.log(tf.math.sqrt(2*np.pi)) - tf.math.reduce_sum(tf.math.log(stds), axis=-1) - 1/2 * tf.math.reduce_sum(tf.math.square(encodings_true - encodings_pred[..., 0]) / tf.math.square(stds), axis=-1)

def gaussian_log_likelihood_unit_variance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log-likelihood of the true encodings with unit variance (used during warm-up)

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]; only the means are used

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*tf.math.log(tf.math.sqrt(2*np.pi)) - 1/2 * tf.math.reduce_sum(tf.math.square(encodings_true - encodings_pred[..., 0]), axis=-1)

def gaussian_exp_norm_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood divided by its maximum over all inputs, computed in log space (used to plot densities)

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood divided by the maximum likelihood, shape [...]
    """

    loglikelihood  = gaussian_log_likelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = tf.math.exp(loglikelihood)

    return likelihood

def laplace_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log-likelihood of the true encodings, summed over the encoding dimensions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    scales = tf.math.maximum(encodings_pred[..., 1], MIN_SCALE)

    return -encodings_pred.shape[-2]*tf.math.log(2.0) - tf.math.reduce_sum(tf.math.log(scales), axis=-1) - tf.math.reduce_sum(tf.math.abs(encodings_true - encodings_pred[..., 0]) / scales, axis=-1)

def laplace_log_likelihood_unit_scale(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log-likelihood of the true encodings with unit scale (used during warm-up)

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]; only the locations are used

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*tf.math.log(2.0) - tf.math.reduce_sum(tf.math.abs(encodings_true - encodings_pred[..., 0]), axis=-1)

def laplace_exp_norm_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace likelihood divided by its maximum over all inputs, computed in log space (used to plot densities)

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood divided by the maximum likelihood, shape [...]
    """

    loglikelihood  = laplace_log_likelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = tf.math.exp(loglikelihood)

    return likelihood

def mahalanobis_distance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Mahalanobis distance of the true encodings to the predicted (diagonal) Gaussian distributions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: Mahalanobis distance, shape [...]
    """

    return tf.math.sqrt(squared_mahalanobis_distance(encodings_true, encodings_pred))

def squared_mahalanobis_distance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Squared Mahalanobis distance of the true encodings to the predicted (diagonal) Gaussian distributions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: squared Mahalanobis distance, shape [...]
    """

    stds = tf.math.maximum(encodings_pred[..., 1], MIN_SCALE)

    return tf.math.reduce_sum(tf.math.square(encodings_true - encodings_pred[..., 0]) / tf.math.square(stds), axis=-1)

def chi_squared_cdf(x: tf.Tensor, df: int) -> tf.Tensor:
    """Cumulative distribution function of the chi-squared distribution

    Args:
        x (tf.Tensor): input values
        df (int): number of degrees of freedom; for tail probabilities, the encoding size Ne

    Returns:
        tf.Tensor: CDF values, same shape as x
    """

    return tfp.distributions.Chi2(df).cdf(x)

def multivariate_gaussian_tail_probability(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Tail probability (probability of conformance) of the true encodings under the predicted Gaussian distributions

    The squared Mahalanobis distance follows a chi-squared distribution with Ne degrees of freedom; the tail
    probability is its survival function, computed in float64 so that small probabilities are not rounded to 0.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: tail probabilities, shape [...]
    """

    # Determine squared Mahalanobis distances
    dist = squared_mahalanobis_distance(encodings_true, encodings_pred)

    # Determine tail probabilities with the survival function of the Chi-squared distribution
    tail_prob = tf.math.igammac(tf.constant(encodings_pred.shape[-2] / 2, tf.float64), tf.cast(dist, tf.float64) / 2)

    return tf.cast(tail_prob, dist.dtype)
