from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics
import scipy.stats
import scipy.optimize
from tqdm import tqdm

def compute_matrices(MNIST_inst, encoder_model, ar_model, forecast_models, batch_size):
    """Compute per-image encodings/forecasts and the resulting pairwise log-likelihood,
    tail-probability and successive-digit truth matrices for a MNIST split."""

    encodings             = encoder_model.predict(MNIST_inst.dataset_with_labels[0], batch_size=batch_size, verbose=1)
    contexts              = ar_model.predict(tf.expand_dims(encodings, axis=1), batch_size=batch_size, verbose=1)[:, 0, :]
    encodings_future_pred = forecast_models[0].predict(contexts, batch_size=batch_size, verbose=1)

    n = len(MNIST_inst.dataset_with_labels[0])
    loglikelihoods = np.empty((n, n), dtype=np.float32)
    probabilities  = np.empty((n, n), dtype=np.float32)

    for i in tqdm(range(n)):
        loglikelihoods[i, :] = PPC.probability.GaussianLogLikelihood(encodings, encodings_future_pred[i:i+1])
        probabilities[i, :]  = PPC.probability.MultiVariateGaussianTailProbability(encodings, encodings_future_pred[i:i+1])

    truth = np.zeros((n, n), dtype=bool)
    for label_a, label_b in itertools.product(range(10), repeat=2):
        if np.mod(label_a + 1, 10) == label_b:
            inds_a = np.argwhere(MNIST_inst.dataset_with_labels[1] == label_a).ravel()
            inds_b = np.argwhere(MNIST_inst.dataset_with_labels[1] == label_b).ravel()
            truth[np.ix_(inds_a, inds_b)] = True

    return encodings, encodings_future_pred, loglikelihoods, probabilities, truth

if  __name__ == '__main__':

    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    models_dir = Path(__file__).parent.resolve() / 'models/'

    val_MNIST_inst  = tools.data.MNIST('val')
    test_MNIST_inst = tools.data.MNIST('test')

    for save_dir in models_dir.iterdir():

        print(save_dir.stem)

        encoder_model, decoder_model, ar_model, forecast_models = PPC.tools.load_ppc_models(save_dir)
        settings = PPC.tools.Settings.load(save_dir / 'settings.json')

        # Pairwise scores on validation (threshold search) and test (figures + final metrics)
        _, _, _, val_probabilities, val_truth = compute_matrices(
            val_MNIST_inst, encoder_model, ar_model, forecast_models, settings.batch_size,
        )
        test_encodings, test_encodings_future_pred, test_loglikelihoods, test_probabilities, test_truth = compute_matrices(
            test_MNIST_inst, encoder_model, ar_model, forecast_models, settings.batch_size,
        )

        # Figure 6: average log-likelihood / probability of conformance per digit pair (test set)
        loglikelihoods_by_labels = np.empty((10, 10), dtype=np.float32)
        probabilities_by_labels  = np.empty((10, 10), dtype=np.float32)

        for label_a, label_b in itertools.product(range(10), repeat=2):
            inds_a = np.argwhere(test_MNIST_inst.dataset_with_labels[1] == label_a).ravel()
            inds_b = np.argwhere(test_MNIST_inst.dataset_with_labels[1] == label_b).ravel()

            loglikelihoods_by_labels[label_a, label_b] = np.mean(test_loglikelihoods[np.ix_(inds_a, inds_b)])
            probabilities_by_labels[label_a, label_b]  = np.mean(test_probabilities[np.ix_(inds_a, inds_b)])

        plt.figure(figsize=[6.85, 3])
        plt.subplot(1, 2, 1)
        plt.title('Log-likelihood')
        plt.imshow(loglikelihoods_by_labels.T, cmap=tools.colormaps.get_modified_oranges())
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.xlabel('First digit')
        plt.ylabel('Second digit')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 2, 2)
        plt.title('Probability of conformance')
        plt.imshow(probabilities_by_labels.T*100, cmap=tools.colormaps.get_modified_oranges(), vmin=0, vmax=100)
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.xlabel('First digit')
        plt.ylabel('Second digit')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_dir / 'mnist-matrix.pdf', transparent=True)
        plt.close()

        # Flip to anomaly-detection framing: positive class = non-successive (anomalous) pair
        val_y_true  = ~val_truth.flatten()
        test_y_true = ~test_truth.flatten()

        val_y_pred  = 1 - val_probabilities.flatten()
        test_y_pred = 1 - test_probabilities.flatten()

        # Figure 7: ROC + PRC curves on the test set
        roc_fpr, roc_tpr, _          = sklearn.metrics.roc_curve(y_true=test_y_true, y_score=test_y_pred)
        prc_precision, prc_recall, _ = sklearn.metrics.precision_recall_curve(y_true=test_y_true, probas_pred=test_y_pred)

        plt.figure(figsize=[6.85, 3])

        ax = plt.subplot(1, 2, 1)
        plt.title('Receiver Operating Characteristic Curve')
        plt.plot(roc_fpr*100, roc_tpr*100, label='PPC-MNIST')
        plt.plot([0, 100], [0, 100], linestyle='--', color='grey', label='Random classifier')
        plt.xlabel('False positive rate [%]')
        plt.ylabel('True positive rate [%]')
        plt.xlim([-3, 103])
        plt.ylim([-3, 103])
        plt.grid(visible=True, which='both')
        plt.legend()

        plt.subplot(1, 2, 2, sharey=ax)
        plt.title('Precision-Recall Curve')
        plt.plot(prc_recall*100, prc_precision*100, label='PPC-MNIST')
        plt.plot([0, 100], [100*np.mean(test_y_true)]*2, linestyle='--', color='grey', label='Random classifier')
        plt.xlabel('Recall [%]')
        plt.ylabel('Precision [%]')
        plt.xlim([-3, 103])
        plt.ylim([-3, 103])
        plt.grid(visible=True, which='both')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_dir / 'mnist-curves.pdf', transparent=True)
        plt.close()

        roc_auc = sklearn.metrics.roc_auc_score(y_true=test_y_true, y_score=test_y_pred)
        prc_auc = sklearn.metrics.average_precision_score(y_true=test_y_true, y_score=test_y_pred)

        print(settings)
        print(f'ROC AUC: {roc_auc}')
        print(f'PRC AUC: {prc_auc}')

        # Find the decision threshold that maximizes F1 on the validation set
        def func(threshold):

            y_pred_binary = val_y_pred >= float(threshold)

            TP = np.sum(np.logical_and(val_y_true, y_pred_binary))
            FP = np.sum(np.logical_and(~val_y_true, y_pred_binary))
            FN = np.sum(np.logical_and(val_y_true, ~y_pred_binary))

            f1_score = (2*TP)/(2*TP+FP+FN)

            return -f1_score

        res = scipy.optimize.differential_evolution(
            func   = func,
            bounds = [(0., 1.)],
        )

        best_threshold = res.x

        print(f'best_threshold: {best_threshold}')

        # Apply the validation-selected threshold to the test set
        y_pred_binary = test_y_pred >= best_threshold

        TP = np.sum(np.logical_and(test_y_true, y_pred_binary))
        FP = np.sum(np.logical_and(~test_y_true, y_pred_binary))
        TN = np.sum(np.logical_and(~test_y_true, ~y_pred_binary))
        FN = np.sum(np.logical_and(test_y_true, ~y_pred_binary))

        tpr_recall_sensitivity = TP/(TP+FN)
        ppv_precision          = TP/(TP+FP)
        tnr_specificity        = TN/(TN+FP)
        balanced_accuracy      = (tpr_recall_sensitivity + tnr_specificity) / 2
        mcc                    = sklearn.metrics.matthews_corrcoef(test_y_true, y_pred_binary)
        f1_score               = (2*TP)/(2*TP+FP+FN)

        print(f'TP: {TP}')
        print(f'FP: {FP}')
        print(f'TN: {TN}')
        print(f'FN: {FN}')
        print(f'TPR/recall/sensitivity: {tpr_recall_sensitivity}')
        print(f'PPV/precision: {ppv_precision}')
        print(f'TNR/specificity: {tnr_specificity}')
        print(f'Balanced accuracy: {balanced_accuracy}')
        print(f'MCC: {mcc}')
        print(f'F1: {f1_score}')

        # Figure 8: PCA-reduced latent space hexbin plots with predicted-distribution contours
        pca = PPC.dimension.PCA(2)
        pca.fit(test_encodings)

        reduced_encodings = pca.transform(test_encodings)
        reduced_encodings_future_pred_means, reduced_encodings_future_pred_covs = pca.transform_gaussians(test_encodings_future_pred)

        example_digits = [2, 3, 9]
        example_inds   = []
        for example_digit in example_digits:
            example_inds.append(np.argwhere(test_MNIST_inst.dataset_with_labels[1] == example_digit).ravel())
        example_inds = np.concatenate(example_inds)

        xlims = np.array([reduced_encodings[example_inds, 0].min(), reduced_encodings[example_inds, 0].max()]) + np.array([-0.1, 0.1]) * (reduced_encodings[example_inds, 0].max() - reduced_encodings[example_inds, 0].min())
        ylims = np.array([reduced_encodings[example_inds, 1].min(), reduced_encodings[example_inds, 1].max()]) + np.array([-0.1, 0.1]) * (reduced_encodings[example_inds, 0].max() - reduced_encodings[example_inds, 1].min())

        global_vmin = np.inf
        global_vmax = -np.inf
        hbs         = []

        plt.figure(figsize=[6.85, 3])

        for i, example_digit in enumerate(example_digits):

            if i == 0:
                ax = plt.subplot(1, len(example_digits), 1)
            else:
                ax = plt.subplot(1, len(example_digits), i+1, sharex=ax, sharey=ax)

            inds_current = np.argwhere(test_MNIST_inst.dataset_with_labels[1] == example_digit).ravel()
            hbs.append(plt.hexbin(
                x        = reduced_encodings[inds_current, 0],
                y        = reduced_encodings[inds_current, 1],
                gridsize = 30,
                extent   = np.concatenate([xlims, ylims]),
                cmap     = tools.colormaps.get_modified_oranges(),
            ))

            vmin = hbs[i].get_array().min()
            vmax = hbs[i].get_array().max()

            if vmin < global_vmin:
                global_vmin = vmin
            if vmax > global_vmax:
                global_vmax = vmax

            inds_prev = np.argwhere(test_MNIST_inst.dataset_with_labels[1] == np.mod(example_digit - 1, 10)).ravel()

            means = np.mean(reduced_encodings_future_pred_means[inds_prev], axis=0)
            covs  = np.mean(reduced_encodings_future_pred_covs[inds_prev], axis=0)

            xgrid = np.linspace(xlims[0], xlims[1], 1000)
            ygrid = np.linspace(ylims[0], ylims[1], 1000)

            d0, d1 = np.meshgrid(xgrid, ygrid)
            pos    = np.dstack((d0, d1))
            rv     = scipy.stats.multivariate_normal(means, covs)
            d2     = rv.pdf(pos)

            plt.contour(d0, d1, d2, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.axis('square')
            plt.title('Digit {}'.format(example_digit))
            plt.grid()

        for hb in hbs:
            hb.set_clim(vmin=global_vmin, vmax=global_vmax)

        plt.tight_layout()
        plt.savefig(save_dir / 'mnist-latent-hexbin.pdf', transparent=True)
        plt.close()
