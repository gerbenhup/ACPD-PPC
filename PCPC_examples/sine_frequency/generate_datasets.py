from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import tools

if __name__ == '__main__':

    # Training data (split into train/validation/test in train.py). PCPC is tested on the test sets of
    # PPC_examples/sine_frequency, so generate those with PPC_examples/sine_frequency/generate_datasets.py.
    if not (Path(__file__).parent / 'data.h5').exists():
        print('Generating dataset')
        dataset = tools.data.ContrastiveSamples.create(
            path                        = Path(__file__).parent / 'data.h5',
            size                        = 1000000,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            Nchi                        = 9,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
            fm                          = 0.25,
            freq_change_rate            = 0.2,
            baseline_wander_range       = [-1, 1],
            baseline_wander_fc          = 0.7,
            baseline_wander_change_rate = 0.5,
            amplitude_range             = [0.5, 2],
            amplitude_fc                = 1,
            amplitude_change_rate       = 1,
            noise_std_range             = [0, 0.2],
            time_shift_range            = [0, 0.75],
            workers                     = 30,
            seed                        = 0,
        )
