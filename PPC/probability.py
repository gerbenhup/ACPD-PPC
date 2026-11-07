import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import math

def GaussianLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood function

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted encoding distributions, shape [..., Ne, 2]. The last channel contains means and standard deviations.

    Returns:
        tf.Tensor: Gaussian likelihood, shape [...]
    """

    return tf.math.reduce_prod((1/(math.sqrt(2*np.pi)*encodings_pred[..., 1]))*math.exp(-tf.math.square(encodings_true - encodings_pred[..., 0])/(2*math.square(encodings_pred[..., 1]))), axis=-1)

def GaussianLikelihoodUnitVariance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood function which assumes unit variance

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted encoding distributions, shape [..., Ne, 2]. Only the means (channel 0) are used.

    Returns:
        tf.Tensor: Gaussian likelihood, shape [...]
    """

    return tf.math.reduce_prod((1/(math.sqrt(2*np.pi)))*math.exp(-tf.math.square(encodings_true - encodings_pred[..., 0])/2), axis=-1)

def GaussianLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log likelihood function

    Args:
        encodings_true (tf.Tensor): true encodings
        encodings_pred (tf.Tensor): predicted encoding distributions

    Returns:
        tf.Tensor: Gaussian log likelihood
    """

    return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - tf.math.reduce_sum(math.log(encodings_pred[..., 1]), axis=-1) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)

def GaussianLogLikelihoodUnitVariance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian log likelihood function which assumes unit variance

    Args:
        encodings_true (tf.Tensor): true encodings
        encodings_pred (tf.Tensor): predicted encoding distributions

    Returns:
        tf.Tensor: Gaussian log likelihood
    """

    return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]), axis=-1)

def GaussianExpNormLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Gaussian likelihood function with exp-log trick for numerical stability

    Args:
        encodings_true (tf.Tensor): true encodings
        encodings_pred (tf.Tensor): predicted encoding distributions

    Returns:
        tf.Tensor: Gaussian likelihood
    """

    loglikelihood  = GaussianLogLikelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = math.exp(loglikelihood)

    return likelihood

def LaplaceLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log likelihood function

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted encoding distributions, shape [..., Ne, 2]. The last channel contains locations and scales.

    Returns:
        tf.Tensor: Laplace log likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(2.0) - tf.math.reduce_sum(math.log(encodings_pred[..., 1]), axis=-1) - tf.math.reduce_sum(math.abs(encodings_true - encodings_pred[..., 0]) / encodings_pred[..., 1], axis=-1)

def LaplaceLogLikelihoodUnitScale(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace log likelihood function which assumes unit scale

    Args:
        encodings_true (tf.Tensor): true encodings, shape [..., Ne]
        encodings_pred (tf.Tensor): predicted encoding distributions, shape [..., Ne, 2]. Only the locations (channel 0) are used.

    Returns:
        tf.Tensor: Laplace log likelihood, shape [...]
    """

    return -encodings_pred.shape[-2]*math.log(2.0) - tf.math.reduce_sum(math.abs(encodings_true - encodings_pred[..., 0]), axis=-1)

def LaplaceExpNormLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Laplace likelihood function with exp-log trick for numerical stability

    Args:
        encodings_true (tf.Tensor): true encodings
        encodings_pred (tf.Tensor): predicted encoding distributions

    Returns:
        tf.Tensor: Laplace likelihood
    """

    loglikelihood  = LaplaceLogLikelihood(encodings_true, encodings_pred)
    loglikelihood -= tf.math.reduce_max(loglikelihood)
    likelihood     = math.exp(loglikelihood)

    return likelihood

def MahalanobisDistance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Mahalanobis distance

    Args:
        encodings_true (tf.Tensor): sample from the multivariate Gaussian distribution, shape [..., Ne]
        encodings_pred (tf.Tensor): multivariate Gaussian distribution parameters, shape [..., Ne, 2]. The last channel contains means and standard deviations.

    Returns:
        tf.Tensor: Mahalanobis distance
    """

    return math.sqrt(tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1))

def SquaredMahalanobisDistance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Mahalanobis distance

    Args:
        z_hat (tf.Tensor): multivariate gaussian distribution parameters, shape [..., Ne, 2]. The last channel contains means and standard deviations.
        z (tf.Tensor): sample from multivariate gaussian distribution

    Returns:
        tf.Tensor: squared Mahalanobis distance
    """

    return tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)

def ChiSquaredCDF(x: tf.Tensor, df: int) -> tf.Tensor:
    """Cumulative distribution function for the Chi-squared distribution.

    Args:
        x (tf.Tensor): input values
        df (int): number of degrees of freedom (in the context of CPCP: the encoding vector size)

    Returns:
        tf.Tensor: CDF values
    """

    return tfp.distributions.Chi2(df).cdf(x)

def MultiVariateGaussianTailProbability(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
    """Calculate the tail probability given a multivariate gaussian distribution and a sample.

    Args:
        z_hat (tf.Tensor): multivariate gaussian distribution parameters, shape [..., Ne, 2]. The last channel contains means and standard deviations.
        z (tf.Tensor): sample from multivariate gaussian distribution

    Returns:
        tf.Tensor: tail probabilities
    """

    # Determine squared Mahalanobis distances
    dist = SquaredMahalanobisDistance(encodings_true, encodings_pred)

    # Determine associated CDF values for the Chi-squared distribution
    cdf = ChiSquaredCDF(dist, encodings_pred.shape[-2])

    # Convert to tail probability and return
    return 1. - cdf
