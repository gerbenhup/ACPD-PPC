from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import tools
import numpy as np
from typing import Tuple
from multiprocessing import Pool
from tqdm import tqdm
import h5py

class SineFrequencyGenerator:
    """Generator of sine wave samples for PCPC: a past signal, its positive continuation and Nchi negative continuations in other frequency bands"""

    def __init__(
        self,
        Np: int,
        Nf: int,
        L: int,
        Nchi: int,
        fs: int,
        fr: Tuple[float, float],
        fb: float,
        fm: float,
        freq_change_rate: float,
        baseline_wander_range: Tuple[float, float],
        baseline_wander_fc: float,
        baseline_wander_change_rate: float,
        amplitude_range: Tuple[float, float],
        amplitude_fc: float,
        amplitude_change_rate: float,
        noise_std_range: Tuple[float, float],
        time_shift_range: Tuple[float, float],
    ):
        """
        Args:
            Np (int): number of past segments
            Nf (int): number of future segments
            L (int): length of a segment
            Nchi (int): number of negative samples
            fs (int): sampling frequency [Hz]
            fr (Tuple[float, float]): range of the frequency bands [Hz]
            fb (float): width of the frequency bands [Hz]
            fm (float): minimum gap between the positive and negative frequency bands [Hz]
            freq_change_rate (float): standard deviation of the frequency random walk
            baseline_wander_range (Tuple[float, float]): bounds of the baseline wander
            baseline_wander_fc (float): cutoff frequency of the baseline wander lowpass filter [Hz]
            baseline_wander_change_rate (float): standard deviation of the baseline wander random walk
            amplitude_range (Tuple[float, float]): bounds of the amplitude
            amplitude_fc (float): cutoff frequency of the amplitude lowpass filter [Hz]
            amplitude_change_rate (float): standard deviation of the amplitude random walk
            noise_std_range (Tuple[float, float]): range of the additive noise standard deviation
            time_shift_range (Tuple[float, float]): range of the shift of the segment boundaries, as a fraction of L
        """

        self.Np                          = Np
        self.Nf                          = Nf
        self.L                           = L
        self.Nchi                        = Nchi
        self.fs                          = fs
        self.fr                          = fr
        self.fb                          = fb
        self.fm                          = fm
        self.freq_change_rate            = freq_change_rate
        self.baseline_wander_range       = baseline_wander_range
        self.baseline_wander_fc          = baseline_wander_fc
        self.baseline_wander_change_rate = baseline_wander_change_rate
        self.amplitude_range             = amplitude_range
        self.amplitude_fc                = amplitude_fc
        self.amplitude_change_rate       = amplitude_change_rate
        self.noise_std_range             = noise_std_range
        self.time_shift_range            = time_shift_range

    def prepare_samples(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one past sample with its positive and negative future samples

        Args:
            *args, **kwargs: unused, present so this method can be passed to `map()` over a dummy iterable

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: past sample [Np, L, 1], positive future sample [Nf, L, 1], negative future samples [Nchi, Nf, L, 1], positive frequency band [2] and negative frequency bands [Nchi, 2]
        """

        # Determine parameters
        time_shift = int(np.floor(self.L * np.random.uniform(self.time_shift_range[0], self.time_shift_range[1])))
        noise_std  = float(np.random.uniform(self.noise_std_range[0], self.noise_std_range[1]))

        # Determine the positive and negative frequency bands. There must be at least a fm Hz gap between the bands.
        f_band_pos  = np.random.uniform(self.fr[0], self.fr[1]-self.fb) + np.array([0, self.fb])
        f_bands_neg = np.empty((self.Nchi, 2), dtype=np.float32)

        for i in range(self.Nchi):
            while True:
                f_band_neg_cand = np.random.uniform(self.fr[0], self.fr[1]-self.fb) + np.array([0, self.fb])

                if f_band_neg_cand[1] < f_band_pos[0] - self.fm or f_band_neg_cand[0] > f_band_pos[1] + self.fm:
                    f_bands_neg[i] = f_band_neg_cand
                    break

        # Create frequency profiles
        f_past       = tools.data.constrained_random_walk([f_band_pos[0]] * (self.L * self.Np + time_shift), [f_band_pos[1]] * (self.L * self.Np + time_shift), self.freq_change_rate)
        f_future_pos = tools.data.constrained_random_walk([f_band_pos[0]] * (self.L * self.Nf - time_shift), [f_band_pos[1]] * (self.L * self.Nf - time_shift), self.freq_change_rate, last_state=f_past[-1])
        f_future_neg = np.stack([tools.data.constrained_random_walk([f_bands_neg[i, 0]] * (self.L * self.Nf - time_shift), [f_bands_neg[i, 1]] * (self.L * self.Nf - time_shift), self.freq_change_rate, last_state=f_past[-1]) for i in range(len(f_bands_neg))], axis=0)

        # Create baseline wander profile
        baseline_wander_profile = tools.data.smoothened_random_walk((self.Np + self.Nf) * self.L, self.baseline_wander_range[0], self.baseline_wander_range[1], self.fs, self.baseline_wander_fc, self.baseline_wander_change_rate)

        # Create amplitude profile
        amplitude_profile = tools.data.smoothened_random_walk((self.Np + self.Nf) * self.L, self.amplitude_range[0], self.amplitude_range[1], self.fs, self.amplitude_fc, self.amplitude_change_rate)

        # Create noise profile
        noise_profile = tools.data.create_noise_profile((self.Np + self.Nf) * self.L, noise_std)

        # Create final signals
        sample_pos   = tools.data.create_sine_profile(np.concatenate([f_past, f_future_pos]), self.fs)
        sample_pos  *= amplitude_profile
        sample_pos  += baseline_wander_profile
        sample_pos  += noise_profile

        samples_neg  = np.stack([tools.data.create_sine_profile(np.concatenate([f_past, x]), self.fs) for x in f_future_neg], axis=0)
        samples_neg *= amplitude_profile
        samples_neg += baseline_wander_profile
        samples_neg += noise_profile

        # Reshape to PCPC output
        sample_past        = sample_pos[:(self.L * self.Np)]
        sample_past        = np.reshape(sample_past, (self.Np, self.L, 1))
        sample_future_pos  = sample_pos[(self.L * self.Np):]
        sample_future_pos  = np.reshape(sample_future_pos, (self.Nf, self.L, 1))
        samples_future_neg = samples_neg[:, (self.L * self.Np):]
        samples_future_neg = np.reshape(samples_future_neg, (self.Nchi, self.Nf, self.L, 1))

        return sample_past, sample_future_pos, samples_future_neg, f_band_pos, f_bands_neg

if __name__ == '__main__':

    # Dataset size and number of parallel workers
    size      = 1000000
    workers   = 30
    generator = SineFrequencyGenerator(**tools.GENERATOR_SETTINGS)

    # Generate samples in parallel
    pool     = Pool(workers)
    map_func = pool.imap_unordered

    collection_past       = []
    collection_future_pos = []
    collection_future_neg = []
    collection_band_pos   = []
    collection_bands_neg  = []

    for sample_past, sample_future_pos, samples_future_neg, f_band_pos, f_bands_neg in tqdm(map_func(generator.prepare_samples, [None]*size), total=size):

        collection_past.append(sample_past)
        collection_future_pos.append(sample_future_pos)
        collection_future_neg.append(samples_future_neg)
        collection_band_pos.append(f_band_pos)
        collection_bands_neg.append(f_bands_neg)

    # Stack samples into arrays
    collection_past       = np.stack(collection_past, axis=0)
    collection_future_pos = np.stack(collection_future_pos, axis=0)
    collection_future_neg = np.stack(collection_future_neg, axis=0)
    collection_band_pos   = np.stack(collection_band_pos)
    collection_bands_neg  = np.stack(collection_bands_neg)

    pool.close()

    # Save dataset
    with h5py.File((Path(__file__).parent / 'data.h5').resolve(), 'w') as f:
        f.create_dataset('collection_past',       data=collection_past)
        f.create_dataset('collection_future_pos', data=collection_future_pos)
        f.create_dataset('collection_future_neg', data=collection_future_neg)
        f.create_dataset('collection_band_pos',   data=collection_band_pos)
        f.create_dataset('collection_bands_neg',  data=collection_bands_neg)