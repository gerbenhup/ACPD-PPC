from typing import Tuple
import numpy as np
import sklearn.decomposition
import sklearn.manifold

class TSNE:
    """Thin wrapper around scikit-learn's t-SNE"""

    def __init__(self, n_components: int, perplexity: float):
        """
        Args:
            n_components (int): dimensionality of the embedded space
            perplexity (float): t-SNE perplexity, related to the number of nearest neighbours considered
        """

        self.n_components = n_components
        self.perplexity   = perplexity
        self.tsne         = sklearn.manifold.TSNE(
            n_components = self.n_components,
            perplexity   = self.perplexity,
        )

    def fit_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Fit t-SNE on the given encodings and return their low-dimensional embedding

        Args:
            encodings (np.ndarray): encodings to embed, shape [N, Ne]

        Returns:
            np.ndarray: embedded encodings, shape [N, n_components]
        """

        return self.tsne.fit_transform(encodings)

class PCA:
    """Thin wrapper around scikit-learn's PCA, extended to also project Gaussian forecast distributions"""

    def __init__(self, n_components: int):
        """
        Args:
            n_components (int): dimensionality of the embedded space
        """

        self.n_components = n_components
        self.pca          = sklearn.decomposition.PCA(self.n_components)

    def fit(self, encodings: np.ndarray) -> None:
        """Fit PCA on the given encodings

        Args:
            encodings (np.ndarray): encodings to fit on, shape [N, Ne]
        """

        self.pca.fit(encodings)

    def fit_transform(self, encodings: np.ndarray) -> np.ndarray:
        """Fit PCA on the given encodings and return their low-dimensional projection

        Args:
            encodings (np.ndarray): encodings to fit on and project, shape [N, Ne]

        Returns:
            np.ndarray: projected encodings, shape [N, n_components]
        """

        return self.pca.fit_transform(encodings)

    def transform(self, encodings: np.ndarray) -> np.ndarray:
        """Project encodings onto the previously fitted principal components

        Args:
            encodings (np.ndarray): encodings to project, shape [N, Ne]

        Returns:
            np.ndarray: projected encodings, shape [N, n_components]
        """

        return self.pca.transform(encodings)

    def transform_gaussians(self, encodings_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project Gaussian forecast distributions onto the previously fitted principal components

        Args:
            encodings_pred (np.ndarray): predicted encoding distributions, shape [N, Ne, 2] (means and standard deviations)

        Returns:
            Tuple[np.ndarray, np.ndarray]: projected means, shape [N, n_components], and projected covariance matrices, shape [N, n_components, n_components]
        """

        reduced_encodings_pred_means = self.pca.transform(encodings_pred[..., 0])
        reduced_encodings_pred_covs  = np.empty((len(encodings_pred), self.n_components, self.n_components), dtype=encodings_pred.dtype)

        for i in range(len(encodings_pred)):
            reduced_encodings_pred_covs[i] = np.matmul(np.matmul(self.pca.components_, np.diagflat(np.square(encodings_pred[i, :, 1]))), self.pca.components_.T)

        return reduced_encodings_pred_means, reduced_encodings_pred_covs
