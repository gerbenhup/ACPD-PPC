"""Likelihoods, distances and tail probabilities of encodings under the predicted forecast distributions.

The forecast distributions are diagonal: every encoding dimension has its own independent Gaussian (or
Laplace) distribution. They are stored as tensors of shape [..., Ne, 2], where channel 0 holds the means
(locations) and channel 1 the standard deviations (scales).
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import math

def gaussian_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood of the true encodings, i.e. the product of the densities over the encoding dimensions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood, shape [...]
    """

    return tf.math.reduce_prod((1/(math.sqrt(2*np.pi)*encodings_pred[..., 1]))*math.exp(-tf.math.square(encodings_true - encodings_pred[..., 0])/(2*math.square(encodings_pred[..., 1]))), axis=-1)

def gaussian_likelihood_unit_variance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood of the true encodings, ignoring the predicted standard deviations (unit variance)

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]; only the means are used

    Returns:
        tf.Tensor: likelihood, shape [...]
    """

    return tf.math.reduce_prod((1/(math.sqrt(2*np.pi)))*math.exp(-tf.math.square(encodings_true - encodings_pred[..., 0])/2), axis=-1)

def gaussian_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log-likelihood of the true encodings, summed over the encoding dimensions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - tf.math.reduce_sum(math.log(encodings_pred[..., 1]), axis=-1) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)

def gaussian_log_likelihood_unit_variance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log-likelihood of the true encodings, ignoring the predicted standard deviations (unit variance)

    Used during the warm-up phase of training, before the standard deviations are learned.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]; only the means are used

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]), axis=-1)

def gaussian_exp_norm_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood relative to its maximum over all inputs, computed in log space for numerical stability

    The log-likelihoods are shifted so that their maximum is 0 before taking the exponent, so the result
    lies in (0, 1]. Useful for plotting the shape of a predicted density, which is normalized afterwards.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood divided by the maximum likelihood, shape [...]
    """

    loglikelihood  = gaussian_log_likelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = math.exp(loglikelihood)

    return likelihood

def laplace_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log-likelihood of the true encodings, summed over the encoding dimensions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(2.0) - tf.math.reduce_sum(math.log(encodings_pred[..., 1]), axis=-1) - tf.math.reduce_sum(math.abs(encodings_true - encodings_pred[..., 0]) / encodings_pred[..., 1], axis=-1)

def laplace_log_likelihood_unit_scale(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log-likelihood of the true encodings, ignoring the predicted scales (unit scale)

    Used during the warm-up phase of training, before the scales are learned.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]; only the locations are used

    Returns:
        tf.Tensor: log-likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(2.0) - tf.math.reduce_sum(math.abs(encodings_true - encodings_pred[..., 0]), axis=-1)

def laplace_exp_norm_log_likelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace likelihood relative to its maximum over all inputs, computed in log space for numerical stability

    See `gaussian_exp_norm_log_likelihood`.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted locations and scales, shape [..., Ne, 2]

    Returns:
        tf.Tensor: likelihood divided by the maximum likelihood, shape [...]
    """

    loglikelihood  = laplace_log_likelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = math.exp(loglikelihood)

    return likelihood

def mahalanobis_distance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Mahalanobis distance of the true encodings to the predicted (diagonal) Gaussian distributions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: Mahalanobis distance, shape [...]
    """

    return math.sqrt(tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1))

def squared_mahalanobis_distance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Squared Mahalanobis distance of the true encodings to the predicted (diagonal) Gaussian distributions

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: squared Mahalanobis distance, shape [...]
    """

    return tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)

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

    The squared Mahalanobis distance of a sample from an Ne-dimensional Gaussian follows a chi-squared
    distribution with Ne degrees of freedom. The tail probability is the probability of a sample lying
    at least as far from the mean: 1 - CDF(squared distance). Low values indicate anomalies.

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted means and standard deviations, shape [..., Ne, 2]

    Returns:
        tf.Tensor: tail probabilities, shape [...]
    """

    # Determine squared Mahalanobis distances
    dist = squared_mahalanobis_distance(encodings_true, encodings_pred)

    # Determine associated CDF values for the Chi-squared distribution
    cdf = chi_squared_cdf(dist, encodings_pred.shape[-2])

    # Convert to tail probability and return
    return 1. - cdf
