from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
import scipy.optimize

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

    test_GridSamples_inst = tools.data.GridSamples(Path(__file__).parent / 'test_dataset.h5')
    val_data_inst         = tools.data.PositiveNegativeSamples(Path(__file__).parent / 'metric_val_dataset.h5')
    test_data_inst        = tools.data.PositiveNegativeSamples(Path(__file__).parent / 'metric_test_dataset.h5')

    # Example segments with increasing frequency deviation across the change point
    freq_dev_examples   = np.array([0, 1, 2])
    example_sample_inds = np.empty(len(freq_dev_examples), dtype=int)

    for i, freq_dev in enumerate(freq_dev_examples):
        candidate_ind = np.argmin(np.abs(np.abs(test_GridSamples_inst.f_center_past - test_GridSamples_inst.f_center_future) - freq_dev))
        example_sample_inds[i] = candidate_ind

    t = np.arange(0, test_GridSamples_inst.flat_data.shape[1]/test_GridSamples_inst.fs, 1/test_GridSamples_inst.fs)

    for save_dir in sorted(models_dir.iterdir()):

        print(save_dir.stem)

        settings = PPC.tools.Settings.load(save_dir / 'settings.json')
        encoder_model, ar_model, forecast_models = PPC.tools.load_pcpc_models(save_dir)

        print(settings)

        encoder_model_td1 = PPC.wrappers.time_distributed(encoder_model)
        forecast_model_ts = PPC.wrappers.time_stacked(forecast_models, [False])

        # Figure: example signal segments
        plt.figure(figsize=[6.85, 4])
        for i, example_ind in enumerate(example_sample_inds):
            if i == 0:
                ax = plt.subplot(len(example_sample_inds), 1, i+1)
                plt.xlim([t[0], t[-1]])
                plt.ylim([test_GridSamples_inst.flat_data[example_sample_inds].min(), test_GridSamples_inst.flat_data[example_sample_inds].max()])
            else:
                ax = plt.subplot(len(example_sample_inds), 1, i+1, sharex=ax, sharey=ax)

            if i < len(example_sample_inds) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel('Time [s]')

            for j, l in enumerate(np.arange(test_GridSamples_inst.L, test_GridSamples_inst.flat_data.shape[1], test_GridSamples_inst.L)):
                if i == 0 and j == 0:
                    plt.axvline(t[l], color='red', alpha=0.8, linestyle='--', label='Segment delineation')
                else:
                    plt.axvline(t[l], color='red', alpha=0.8, linestyle='--')

            plt.plot(t, test_GridSamples_inst.flat_data[example_ind])

            if i == 0:
                plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_dir / 'sine-examples.pdf', transparent=True)
        plt.close()

        # Generate encodings over the frequency grid
        encodings_past        = encoder_model_td1.predict(test_GridSamples_inst.past_data, batch_size=settings.batch_size, verbose=1)
        encodings_future_true = encoder_model_td1.predict(test_GridSamples_inst.future_data, batch_size=settings.batch_size, verbose=1)

        context_past          = ar_model.predict(encodings_past, batch_size=settings.batch_size, verbose=1)[:, -1, :]
        encodings_future_pred = forecast_model_ts(context_past)

        encodings_future_true = encodings_future_true[:, 0, :]
        encodings_future_pred = encodings_future_pred[:, 0, :]

        loglikelihoods = PPC.probability.gaussian_log_likelihood(encodings_future_true, encodings_future_pred)
        probabilities  = PPC.probability.multivariate_gaussian_tail_probability(encodings_future_true, encodings_future_pred)

        loglikelihoods_res = tf.reshape(loglikelihoods, (loglikelihoods.shape[0]//test_GridSamples_inst.iterations, test_GridSamples_inst.iterations))
        probabilities_res  = tf.reshape(probabilities, (probabilities.shape[0]//test_GridSamples_inst.iterations, test_GridSamples_inst.iterations))
        loglikelihoods_agg = np.array(tf.math.reduce_mean(loglikelihoods_res, axis=1))
        probabilities_agg  = np.array(tf.math.reduce_mean(probabilities_res, axis=1))

        # Matrices indexed [past frequency, future frequency], transposed so the past frequency is on the x-axis
        loglikelihoods_mat = np.reshape(loglikelihoods_agg, (test_GridSamples_inst.gridsize, test_GridSamples_inst.gridsize)).T
        probabilities_mat  = np.reshape(probabilities_agg, (test_GridSamples_inst.gridsize, test_GridSamples_inst.gridsize)).T

        # Figure: log-likelihood / probability of conformance heatmap over the frequency grid
        plt.figure(figsize=[6.85, 3])
        plt.subplot(1, 2, 1)
        plt.title('Log-likelihood')
        plt.axis('square')
        plt.pcolormesh(test_GridSamples_inst.grid, test_GridSamples_inst.grid, loglikelihoods_mat, cmap=tools.colormaps.get_modified_oranges(), vmin=np.quantile(loglikelihoods_mat, 0.05), vmax=np.quantile(loglikelihoods_mat, 0.95))
        plt.plot(test_GridSamples_inst.grid, test_GridSamples_inst.grid-0.5*test_GridSamples_inst.fb, color='tab:blue', alpha=0.8, linestyle='--', linewidth=1, label='Frequency\nband margin')
        plt.plot(test_GridSamples_inst.grid, test_GridSamples_inst.grid+0.5*test_GridSamples_inst.fb, color='tab:blue', alpha=0.8, linestyle='--', linewidth=1)
        plt.xlim([test_GridSamples_inst.grid.min(), test_GridSamples_inst.grid.max()])
        plt.ylim([test_GridSamples_inst.grid.min(), test_GridSamples_inst.grid.max()])
        plt.xlabel('$f_c$ before change point [Hz]')
        plt.ylabel('$f_c$ after change point [Hz]')
        plt.legend()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 2, 2)
        plt.title('Probability of conformance')
        plt.axis('square')
        plt.pcolormesh(test_GridSamples_inst.grid, test_GridSamples_inst.grid, probabilities_mat*100, cmap=tools.colormaps.get_modified_oranges(), vmin=0, vmax=100)
        plt.plot(test_GridSamples_inst.grid, test_GridSamples_inst.grid-0.5*test_GridSamples_inst.fb, color='tab:blue', alpha=0.8, linestyle='--', linewidth=1, label='Frequency\nband margin')
        plt.plot(test_GridSamples_inst.grid, test_GridSamples_inst.grid+0.5*test_GridSamples_inst.fb, color='tab:blue', alpha=0.8, linestyle='--', linewidth=1)
        plt.xlim([test_GridSamples_inst.grid.min(), test_GridSamples_inst.grid.max()])
        plt.ylim([test_GridSamples_inst.grid.min(), test_GridSamples_inst.grid.max()])
        plt.xlabel('$f_c$ before change point [Hz]')
        plt.ylabel('$f_c$ after change point [Hz]')
        plt.legend()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_dir / 'sine-matrix.pdf', transparent=False)
        plt.close()

        # Generate encodings for the labelled val/test anomaly-detection sets
        val_encodings_past  = encoder_model_td1.predict(val_data_inst.samples[:, :5, :, :], batch_size=settings.batch_size, verbose=1)
        test_encodings_past = encoder_model_td1.predict(test_data_inst.samples[:, :5, :, :], batch_size=settings.batch_size, verbose=1)

        val_encodings_future_true  = encoder_model_td1.predict(val_data_inst.samples[:, 5:, :, :], batch_size=settings.batch_size, verbose=1)
        test_encodings_future_true = encoder_model_td1.predict(test_data_inst.samples[:, 5:, :, :], batch_size=settings.batch_size, verbose=1)

        val_context_past  = ar_model.predict(val_encodings_past, batch_size=settings.batch_size, verbose=1)[:, -1, :]
        test_context_past = ar_model.predict(test_encodings_past, batch_size=settings.batch_size, verbose=1)[:, -1, :]

        val_encodings_future_pred  = forecast_model_ts(val_context_past)
        test_encodings_future_pred = forecast_model_ts(test_context_past)

        val_encodings_future_true  = val_encodings_future_true[:, 0, :]
        test_encodings_future_true = test_encodings_future_true[:, 0, :]

        val_encodings_future_pred  = val_encodings_future_pred[:, 0, :]
        test_encodings_future_pred = test_encodings_future_pred[:, 0, :]

        val_probabilities  = PPC.probability.multivariate_gaussian_tail_probability(val_encodings_future_true, val_encodings_future_pred)
        test_probabilities = PPC.probability.multivariate_gaussian_tail_probability(test_encodings_future_true, test_encodings_future_pred)

        val_y_true  = val_data_inst.anomaly_labels.astype(bool)
        test_y_true = test_data_inst.anomaly_labels.astype(bool)

        val_y_pred  = 1 - val_probabilities
        test_y_pred = 1 - test_probabilities

        # Figure: ROC + PRC curves on the test set
        roc_fpr, roc_tpr, _          = sklearn.metrics.roc_curve(y_true=test_y_true, y_score=test_y_pred)
        prc_precision, prc_recall, _ = sklearn.metrics.precision_recall_curve(y_true=test_y_true, probas_pred=test_y_pred)

        plt.figure(figsize=[6.85, 3])

        ax = plt.subplot(1, 2, 1)
        plt.title('Receiver Operating Characteristic Curve')
        plt.plot(roc_fpr*100, roc_tpr*100, label='PCPC-SINE')
        plt.plot([0, 100], [0, 100], linestyle='--', color='grey', label='Random classifier')
        plt.xlabel('False positive rate [%]')
        plt.ylabel('True positive rate [%]')
        plt.xlim([-3, 103])
        plt.ylim([-3, 103])
        plt.grid(visible=True, which='both')
        plt.legend()

        plt.subplot(1, 2, 2, sharey=ax)
        plt.title('Precision-Recall Curve')
        plt.plot(prc_recall*100, prc_precision*100, label='PCPC-SINE')
        plt.plot([0, 100], [100*np.mean(test_y_true)]*2, linestyle='--', color='grey', label='Random classifier')
        plt.xlabel('Recall [%]')
        plt.ylabel('Precision [%]')
        plt.xlim([-3, 103])
        plt.ylim([-3, 103])
        plt.grid(visible=True, which='both')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_dir / 'sine-curves.pdf', transparent=True)
        plt.close()

        roc_auc = sklearn.metrics.roc_auc_score(y_true=test_y_true, y_score=test_y_pred)
        prc_auc = sklearn.metrics.average_precision_score(y_true=test_y_true, y_score=test_y_pred)

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
            seed   = 0,
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
