from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

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

    # Load generator settings, used to generate the test signals
    settings = PPC.tools.Settings(**tools.GENERATOR_SETTINGS)

    # Load models
    encoder_model, ar_model, forecast_models = PPC.tools.load_pcpc_models(
        dir = save_dir,
    )

    # Settings for frequency sweep
    resolution = 25 # steps/Hz
    iterations = 100
    workers    = 20

    # Initiate anomaly detector
    anomaly = PPC.Anomaly(
        encoder_model   = encoder_model,
        ar_model        = ar_model,
        forecast_models = forecast_models,
    )

    # Determine frequency grid
    freqs = np.arange(settings.fr[0]+0.5*settings.fb, settings.fr[1]-0.5*settings.fb, 1/resolution)
    probs = np.empty((len(freqs), len(freqs), iterations), dtype=np.float32)

    # Set-up pool of workers
    pool = Pool(workers)

    # Set-up progress bars
    pbar = tqdm(total=len(freqs)*len(freqs)*iterations)

    # Loop through all combinations of past and future frequencies
    for i, f_center_past in enumerate(freqs):

        for j, f_center_future in enumerate(freqs):

            signals_batch = []

            # Collect number of iterations using pool of workers
            for signal_reshaped in pool.imap_unordered(tools.data.generate_signal_mp, [[f_center_past, f_center_future, settings]]*iterations):
                signals_batch.append(signal_reshaped)
                pbar.update()

            # Concatenate signals and apply anomaly detector on it, taking the tail probability of the first future segment
            signals_batch  = np.concatenate(signals_batch, axis=0)
            probs[i, j, :] = np.array(anomaly(signals_batch))[:, settings.Np, 0]

    pool.close()
    pbar.close()

    # Plot results. probs is indexed [past, future], so transpose to put the past frequency on the x-axis
    probs_mean = np.mean(probs, axis=2).T

    plt.figure(figsize=[6, 5])
    plt.title('Probability of a signal segment\nbeing normal after a change point')
    plt.pcolormesh(freqs, freqs, probs_mean*100, vmin=0, vmax=100)
    plt.plot(freqs, freqs-settings.fb, color='red', alpha=0.8, linestyle='--', label='Frequency band margin')
    plt.plot(freqs, freqs+settings.fb, color='red', alpha=0.8, linestyle='--')
    plt.xlim([freqs[0], freqs[-1]])
    plt.ylim([freqs[0], freqs[-1]])
    plt.xlabel('Center frequency before change point [Hz]')
    plt.ylabel('Center frequency after change point [Hz]')
    plt.legend()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir / 'pcpc-sine-matrix.pdf', transparent=True)
    plt.close()
