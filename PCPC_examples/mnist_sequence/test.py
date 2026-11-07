from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import scipy.stats

if  __name__ == '__main__':

    # Select GPU's
    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    # Select save directory
    save_dir = Path(__file__).parent.resolve() / 'models/model-001/'

    # Load settings
    settings = PPC.tools.Settings.load(save_dir / 'settings.json')

    # Load models
    encoder_model, ar_model, forecast_models = PPC.tools.load_pcpc_models(
        dir = save_dir,
    )

    # Load MNIST data and preprocess
    _, (x, y) = tf.keras.datasets.mnist.load_data()
    x         = np.expand_dims((x / 255).astype(np.float32), axis=-1)

    # Sort data on digit values
    sort_inds = np.argsort(y)
    x         = x[sort_inds]
    y         = y[sort_inds]

    # Generate empty score tables
    loglikelihoods = np.empty((len(x), len(x)), dtype=np.float32)
    probabilities  = np.empty((len(x), len(x)), dtype=np.float32)

    # Generate encodings for all images
    encodings             = encoder_model.predict(x, batch_size=settings.batch_size)
    contexts              = ar_model.predict(tf.expand_dims(encodings, axis=1), batch_size=settings.batch_size)[:, 0, :]
    encodings_future_pred = forecast_models[0].predict(contexts, batch_size=settings.batch_size)

    # Plot t-SNE reduced encodings
    reduced_encodings = PPC.dimension.TSNE(2, 30).fit_transform(encodings)

    plt.figure()
    for unique_label in np.unique(y):
        inds = np.argwhere(y == unique_label).ravel()
        plt.scatter(reduced_encodings[inds, 0], reduced_encodings[inds, 1], marker='.', label='Digit {}'.format(unique_label), alpha=0.5)
    plt.legend(labelspacing=0.2, handletextpad=0.2)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE reduced latent space\nrepresentation of MNIST samples')
    plt.savefig(save_dir / 'pcpc-mnist-tsne.pdf', transparent=True)
    plt.close()

    # Generate stats for all combinations and plot them
    for i in tqdm(range(len(x))):
        loglikelihoods[i, :] = PPC.probability.GaussianLogLikelihood(encodings, encodings_future_pred[i:i+1])
        probabilities[i, :]  = PPC.probability.MultiVariateGaussianTailProbability(encodings, encodings_future_pred[i:i+1])

    plt.figure()
    plt.imshow(loglikelihoods, cmap='Oranges')
    plt.xticks([])
    plt.yticks([])
    plt.title('Log-likelihood per sample combination')
    plt.colorbar()
    plt.savefig(save_dir / 'pcpc-mnist-loglikelihood-samples.png', dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(probabilities, cmap='Oranges')
    plt.xticks([])
    plt.yticks([])
    plt.title('Probability of conformity per sample combination')
    plt.colorbar()
    plt.savefig(save_dir / 'pcpc-mnist-probability-samples.png', dpi=300)
    plt.close()

    # Summarize statistics and plot them
    loglikelihoods_by_label = np.empty((10, 10), dtype=np.float32)
    probabilities_by_label  = np.empty((10, 10), dtype=np.float32)

    for label_a, label_b in tqdm(list(itertools.product(range(10), repeat=2))):

        inds_a = np.argwhere(y == label_a).ravel()
        inds_b = np.argwhere(y == label_b).ravel()

        loglikelihoods_by_label[label_a, label_b] = np.mean(loglikelihoods[np.ix_(inds_a, inds_b)])
        probabilities_by_label[label_a, label_b]  = np.mean(probabilities[np.ix_(inds_a, inds_b)])

    plt.figure()
    plt.imshow(loglikelihoods_by_label, cmap='Oranges')
    plt.title('Log-likelihood per digit combination')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.colorbar()
    plt.savefig(save_dir / 'pcpc-mnist-loglikelihood-digits.pdf', transparent=True)
    plt.close()

    plt.figure()
    plt.imshow(probabilities_by_label, cmap='Oranges')
    plt.title('Probability of conformity per digit combination')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.colorbar()
    plt.savefig(save_dir / 'pcpc-mnist-probability-digits.pdf', transparent=True)
    plt.close()

    # Create PCA reduced encodings, make a prediction of the next digit for each digit
    # and plot reduced prediction distributions over the latent space
    pca                   = PPC.dimension.PCA(2)
    reduced_encodings_pca = pca.fit_transform(encodings)

    for digit in range(10):

        example_ind = np.argwhere(y == digit)[0, 0]

        reduced_encodings_future_pred_mu, reduced_encodings_future_pred_cov = pca.transform_gaussians(encodings_future_pred[example_ind:example_ind+1])
        reduced_encodings_future_pred_mu  = reduced_encodings_future_pred_mu[0]
        reduced_encodings_future_pred_cov = reduced_encodings_future_pred_cov[0]

        d0, d1 = np.mgrid[reduced_encodings_pca[:, 0].min():reduced_encodings_pca[:, 0].max():0.01, reduced_encodings_pca[:, 1].min():reduced_encodings_pca[:, 1].max():0.01]
        pos    = np.dstack((d0, d1))
        rv     = scipy.stats.multivariate_normal(reduced_encodings_future_pred_mu, reduced_encodings_future_pred_cov)
        d2     = rv.pdf(pos)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        fig.suptitle('PCA reduced latent space representation of\nMNIST samples with prediction for digit {}'.format(y[example_ind]))

        for unique_label in np.unique(y):
            inds = np.argwhere(y == unique_label).ravel()
            ax1.scatter(reduced_encodings_pca[inds, 0], reduced_encodings_pca[inds, 1], marker='.', label='Digit {}'.format(unique_label), alpha=0.5)

        ax1.contour(d0, d1, d2)
        ax1.legend(labelspacing=0.2, handletextpad=0.2)
        ax1.set_xticks([])
        ax1.set_yticks([])

        for unique_label in np.unique(y):
            if unique_label == np.mod(y[example_ind]+1, 10):
                inds = np.argwhere(y == np.mod(y[example_ind]+1, 10)).ravel()
                ax2.scatter(reduced_encodings_pca[inds, 0], reduced_encodings_pca[inds, 1], marker='.', label='Digit {}'.format(unique_label), alpha=0.5)
            else:
                # Plot nothing, but advance the color cycle so the digit keeps its color from ax1
                ax2.scatter([np.nan], [np.nan])

        ax2.contour(d0, d1, d2)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(save_dir / 'pcpc-mnist-pca-digit-{}.pdf'.format(digit), transparent=True)
        plt.close()
