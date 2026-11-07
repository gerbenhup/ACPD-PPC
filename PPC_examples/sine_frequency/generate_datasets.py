from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import tools

if __name__ == '__main__':

    if not (Path(__file__).parent / 'train_dataset.h5').exists():
        print('Generating train dataset')
        dataset = tools.data.PositiveSamples.create(
            path                        = Path(__file__).parent / 'train_dataset.h5',
            size                        = 1000000,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
            freq_change_rate            = 0.2,
            baseline_wander_range       = [-1, 1],
            baseline_wander_fc          = 0.7,
            baseline_wander_change_rate = 0.5,
            amplitude_range             = [0.5, 2],
            amplitude_fc                = 1,
            amplitude_change_rate       = 1,
            noise_std_range             = [0, 0.2],
            time_shift_range            = [0, 0.75],
            workers                     = 16,
            seed                        = 0,
        )

    if not (Path(__file__).parent / 'val_dataset.h5').exists():
        print('Generating val dataset')
        dataset = tools.data.PositiveSamples.create(
            path                        = Path(__file__).parent / 'val_dataset.h5',
            size                        = 100000,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
            freq_change_rate            = 0.2,
            baseline_wander_range       = [-1, 1],
            baseline_wander_fc          = 0.7,
            baseline_wander_change_rate = 0.5,
            amplitude_range             = [0.5, 2],
            amplitude_fc                = 1,
            amplitude_change_rate       = 1,
            noise_std_range             = [0, 0.2],
            time_shift_range            = [0, 0.75],
            workers                     = 16,
            seed                        = 1,
        )

    if not (Path(__file__).parent / 'test_dataset.h5').exists():
        print('Generating test dataset')
        dataset = tools.data.GridSamples.create(
            path                        = Path(__file__).parent / 'test_dataset.h5',
            gridsize                    = 190,
            iterations                  = 10,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
            freq_change_rate            = 0.2,
            baseline_wander_range       = [-1, 1],
            baseline_wander_fc          = 0.7,
            baseline_wander_change_rate = 0.5,
            amplitude_range             = [0.5, 2],
            amplitude_fc                = 1,
            amplitude_change_rate       = 1,
            noise_std_range             = [0, 0.2],
            time_shift_range            = [0, 0.75],
            workers                     = 16,
            seed                        = 2,
        )

    if not (Path(__file__).parent / 'metric_val_dataset.h5').exists():
        print('Generating metric_val dataset')
        dataset = tools.data.PositiveNegativeSamples.create(
            path                        = Path(__file__).parent / 'metric_val_dataset.h5',
            num_pos                     = 100000,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
            freq_change_rate            = 0.2,
            baseline_wander_range       = [-1, 1],
            baseline_wander_fc          = 0.7,
            baseline_wander_change_rate = 0.5,
            amplitude_range             = [0.5, 2],
            amplitude_fc                = 1,
            amplitude_change_rate       = 1,
            noise_std_range             = [0, 0.2],
            time_shift_range            = [0, 0.75],
            workers                     = 16,
            seed                        = 3,
        )

    if not (Path(__file__).parent / 'metric_test_dataset.h5').exists():
        print('Generating metric_test dataset')
        dataset = tools.data.PositiveNegativeSamples.create(
            path                        = Path(__file__).parent / 'metric_test_dataset.h5',
            num_pos                     = 100000,
            Np                          = 5,
            Nf                          = 3,
            L                           = 256,
            fs                          = 128,
            fr                          = [0.5, 10],
            fb                          = 0.25,
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
            seed                        = 4,
        )
