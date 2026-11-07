from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import gc
import tensorflow as tf

if __name__ == '__main__':

    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    models_dir = Path(__file__).parent.resolve() / 'models/'

    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = {}

    for settings_path in sorted(models_dir.rglob('settings.json')):

        save_dir = settings_path.parent

        if len(list(save_dir.iterdir())) <= 1:
            continue

        print(save_dir.relative_to(models_dir))

        settings = PPC.tools.Settings.load(settings_path)

        # Load models
        encoder_model, decoder_model, ar_model, forecast_models = PPC.tools.load_ppc_models(
            dir = save_dir,
        )
        encoder_model_t1d = PPC.wrappers.time_distributed(encoder_model)
        forecast_model_ts = PPC.wrappers.time_stacked(forecast_models, [False])

        # Create grid
        grid_min = np.min(settings.choice_step_one_values) - 6*(settings.normal_a*np.min(settings.choice_step_one_values)+settings.normal_b)
        grid_max = np.max(settings.choice_step_one_values) + 6*(settings.normal_a*np.max(settings.choice_step_one_values)+settings.normal_b)
        grid_res = 0.01
        grid     = np.arange(grid_min, grid_max+grid_res, grid_res)

        PDF_true = np.empty((len(grid), len(settings.choice_step_one_values)), dtype=np.float32)
        EDF_pred = np.empty((len(grid), len(settings.choice_step_one_values)), dtype=np.float32)

        parameters_true = np.empty((2, len(settings.choice_step_one_values)), dtype=np.float32)

        metrics[save_dir.stem] = []

        # Loop through choice step one values
        for i, choice_step_one_value in enumerate(settings.choice_step_one_values):

            PDF_true[:, i] = tools.functions.gaussian(grid, choice_step_one_value, settings.normal_a*choice_step_one_value+settings.normal_b)

            mu_true, sigma_true = choice_step_one_value, settings.normal_a*choice_step_one_value+settings.normal_b
            parameters_true[:, i] = mu_true, sigma_true

            samples_past   = np.reshape(np.repeat(choice_step_one_value, len(grid)), (len(grid), 1, 1))
            samples_future = np.reshape(grid, (len(grid), 1, 1))

            encodings_past        = encoder_model_t1d.predict(samples_past, batch_size=settings.batch_size, verbose=0)
            encodings_future_pred = forecast_model_ts.predict(ar_model(encodings_past)[:, -1, :], batch_size=settings.batch_size, verbose=0)
            encodings_future_true = encoder_model_t1d.predict(samples_future, batch_size=settings.batch_size, verbose=0)

            EDF_pred[:, i]  = np.array(PPC.probability.gaussian_exp_norm_log_likelihood(encodings_future_true, encodings_future_pred)[:, 0])
            EDF_pred[:, i] /= scipy.integrate.trapezoid(EDF_pred[:, i], grid)

            mu_pred, sigma_pred, mae = tools.functions.fit_gaussian(grid, EDF_pred[:, i])

            metrics[save_dir.stem].extend([mu_true, mu_pred, sigma_true, sigma_pred, mae])

        # Figure 2: ground truth vs. estimated PDFs for each x1 value
        plt.figure(figsize=[6.85, 3])
        for i in range(len(settings.choice_step_one_values)):
            plt.plot(grid, PDF_true[:, i], color=colors[i], linewidth=2.5, alpha=0.5, label=r'Ground truth PDF for $\mu_{{X_2}}={:.0f}$'.format(parameters_true[0, i]))
            plt.plot(grid, EDF_pred[:, i], color=colors[i], linestyle='--', label=r'Estimated PDF for $\mu_{{X_2}}={:.0f}$'.format(parameters_true[0, i]))
        plt.xlim([grid.min(), grid.max()])
        plt.xlabel(r'$x_2$')
        plt.ylabel('Density')
        plt.grid()
        plt.legend(labelspacing=0)
        plt.tight_layout()
        plt.savefig(save_dir / 'proptest.pdf', transparent=True)
        plt.close()

        # Free up memory
        del encoder_model
        del decoder_model
        del ar_model
        del forecast_models
        del encoder_model_t1d
        del forecast_model_ts

        gc.collect()
        tf.keras.backend.clear_session()

    # Table 1: true vs. fitted Gaussian parameters, per model and aggregated (mean ± std)
    for model_name, values in metrics.items():
        print('{}\t'.format(model_name), end='')
        for value in values:
            print('{:.4f}\t'.format(value), end='')
        print()

    metrics_array = np.array(list(metrics.values()))
    metrics_means = np.mean(metrics_array, axis=0)
    metrics_stds  = np.std(metrics_array, axis=0)

    for mean, std in zip(metrics_means, metrics_stds):
        print('{:.2f} ± {:.2f}\t'.format(mean, std), end='')
    print()
