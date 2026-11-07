from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Union, Optional, Tuple, Iterator
import scipy.signal
import scipy.integrate
from multiprocessing import Pool
from tqdm import tqdm
import h5py
import matplotlib as mpl

# Sine wave generator settings, as used in generate_datasets.py
GENERATOR_SETTINGS = dict(
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
)

class models:
    """Encoder, autoregressive and forecast models (identical to the PPC sine example, without decoder)"""

    def encoder_model(Ne: int, filters: int, L: int) -> tf.keras.Model:
        """Create the encoder: two blocks of Conv1D (kernel size 3, ReLU), batch normalization and max pooling,
        with filters and 2*filters channels, and a linear dense layer, [B, L, 1] -> [B, Ne]

        Args:
            Ne (int): size of encoding vector
            filters (int): number of filters in the first convolutional layers
            L (int): length of segment

        Returns:
            tf.keras.Model: TensorFlow encoder model
        """

        inputs = tf.keras.Input((L, 1))

        x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Conv1D(filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(Ne, activation='linear')(x)

        model = tf.keras.Model(inputs, x)

        return model

    def ar_model(Np: int, Ne: int, Ng: int) -> tf.keras.Model:
        """Create the autoregressive model: a GRU with Ng units and linear activation, [B, Np, Ne] -> [B, Np, Ng]

        Args:
            Np (int): number of input encodings
            Ne (int): size of encoding vector
            Ng (int): size of context vector

        Returns:
            tf.keras.Model: TensorFlow autoregressive model
        """

        inputs  = tf.keras.Input((Np, Ne))
        outputs = tf.keras.layers.GRU(Ng, return_sequences=True, activation=None, unroll=True)(inputs)
        model   = tf.keras.Model(inputs, outputs)

        return model

    def forecast_model(Ng: int, Ne: int, dense_units: int) -> tf.keras.Model:
        """Create a forecast model: six ReLU dense layers (dense_units, dense_units, 2*dense_units, 2*dense_units,
        4*dense_units, 4*dense_units units) and linear output layers for the means and log standard deviations,
        [B, Ng] -> [B, Ne, 2]

        Args:
            Ng (int): size of context vector
            Ne (int): size of encoding vector
            dense_units (int): number of dense units after first fully-connected layer

        Returns:
            tf.keras.Model: TensorFlow forecast model
        """

        inputs = tf.keras.Input((Ng,))

        x = tf.keras.layers.Dense(dense_units, activation='relu')(inputs)
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='relu')(x)

        outputs_means = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Dense(Ne, activation='linear')(x)
        # Standard deviations, with clipped log standard deviations to avoid underflow and overflow
        outputs_stds  = tf.keras.layers.Lambda(lambda y: tf.math.exp(tf.clip_by_value(y, -20.0, 88.0)), name='Exponential')(outputs_stds)

        outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='Stack1D')([outputs_means, outputs_stds])

        model = tf.keras.Model(inputs, outputs)

        return model

class data:
    """Sine wave signal generation and the datasets built from it"""

    def constrained_random_walk(lower_bounds: Union[list, np.ndarray], upper_bounds: Union[list, np.ndarray], change_rate: float, last_state: Optional[float]=None, dtype: np.dtype=np.float32) -> np.ndarray:
        """Generate a random walk constrained to a set of lower and upper bounds.

        Args:
            lower_bounds (Union[list, np.ndarray]): List of lower bounds. Must have the same length as the desired random walk signal.
            upper_bounds (Union[list, np.ndarray]): List of upper bounds. Must have the same length as the desired random walk signal.
            change_rate (float): Standard deviation of the random walk.
            last_state (Optional[float], optional): Last state of a previous random walk. Can be used to continue with another random walk. Defaults to None.
            dtype (np.dtype, optional): Type of returned array. Defaults to np.float32.

        Returns:
            np.ndarray: Generated constrained random walk
        """
        x    = np.empty(len(lower_bounds), dtype=dtype)
        x[0] = np.random.uniform(lower_bounds[0], upper_bounds[0]) if last_state is None else last_state

        for i, lower_bound, upper_bound in zip(range(1, len(lower_bounds)), lower_bounds[1:], upper_bounds[1:]):

            x[i] = np.random.normal(loc=x[i-1], scale=change_rate)
            x[i] = np.minimum(x[i], upper_bound)
            x[i] = np.maximum(x[i], lower_bound)

        return x

    def create_sine_profile(f: Union[list, np.ndarray], fs: int) -> np.ndarray:
        """Create a sine wave signal with a wave frequency defined for every sample.

        Args:
            f (Union[list, np.ndarray]): Wave frequency for every sample. Must have the same length as the desired output signal.
            fs (int): Sampling frequency of the output signal.

        Returns:
            np.ndarray: Sine wave output signal.
        """
        return np.sin(2*np.pi*scipy.integrate.cumulative_trapezoid(f, dx=1/fs, initial=0))

    def smoothened_random_walk(size: int, lower_bound: float, upper_bound: float, fs: int, fc: float, change_rate: float, dtype: np.dtype=np.float32) -> np.ndarray:
        """Generate a smoothened constrained random walk

        Args:
            size (int): Desired length of the signal
            lower_bound (float): Lower bound of the signal. In practice, this lower bound is rarely encountered.
            upper_bound (float): Upper bound of the signal. In practice, this lower bound is rarely encountered.
            fs (int): Sampling frequency of the output signal
            fc (float): Cutoff frequency for the lowpass smoothening filter
            change_rate (float): Standard deviation for the random walk.
            dtype (np.dtype, optional): Type of output array. Defaults to np.float32.

        Returns:
            np.ndarray: Smoothened random walk signal
        """

        # Generate list of lower bounds
        lower_bounds = [lower_bound]*size
        upper_bounds = [upper_bound]*size

        # Generate filter coefficients for a lowpass filter
        sos = scipy.signal.butter(4, fc, output='sos', fs=fs)

        # Generate random walk with specified parameters and apply lowpass filter
        baseline_wander = data.constrained_random_walk(lower_bounds, upper_bounds, change_rate, dtype=dtype)
        baseline_wander = scipy.signal.sosfiltfilt(sos, baseline_wander)

        return baseline_wander

    def create_noise_profile(size: int, stddev: float, dtype: np.dtype=np.float32) -> np.ndarray:
        """Create additive Gaussian noise.

        Args:
            size (int): Length of desired noise signal.
            stddev (float): Standard deviation of noise.
            dtype (np.dtype, optional): Type of output noise signal. Defaults to np.float32.

        Returns:
            np.ndarray: Noise signal.
        """
        return np.random.normal(0, stddev, size).astype(dtype)

    def generate_signal(
        f_center_past,
        f_center_future,
        L,
        time_shift_range,
        noise_std_range,
        fb,
        Np,
        Nf,
        freq_change_rate,
        baseline_wander_range,
        baseline_wander_fc,
        baseline_wander_change_rate,
        amplitude_range,
        amplitude_fc,
        amplitude_change_rate,
        fs,
    ):
        """Generate a single sine wave signal whose frequency band changes from a past to a future center frequency

        Args:
            f_center_past (float): center frequency of the band before the change point [Hz]
            f_center_future (float): center frequency of the band after the change point [Hz]
            L (int): length of a segment
            time_shift_range (Tuple[float, float]): range of the change point shift, as a fraction of L
            noise_std_range (Tuple[float, float]): range of the additive noise standard deviation
            fb (float): width of the frequency bands [Hz]
            Np (int): number of past segments
            Nf (int): number of future segments
            freq_change_rate (float): standard deviation of the frequency random walk
            baseline_wander_range (Tuple[float, float]): bounds of the baseline wander
            baseline_wander_fc (float): cutoff frequency of the baseline wander lowpass filter [Hz]
            baseline_wander_change_rate (float): standard deviation of the baseline wander random walk
            amplitude_range (Tuple[float, float]): bounds of the amplitude
            amplitude_fc (float): cutoff frequency of the amplitude lowpass filter [Hz]
            amplitude_change_rate (float): standard deviation of the amplitude random walk
            fs (int): sampling frequency [Hz]

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: time vector, signal, and signal reshaped to segments of shape [1, Np+Nf, L, 1]
        """

        # Determine parameters
        time_shift = int(np.floor(L * np.random.uniform(time_shift_range[0], time_shift_range[1])))
        noise_std  = float(np.random.uniform(noise_std_range[0], noise_std_range[1]))

        # Determine positive and negative frequency bands
        f_band_past   = np.array([-0.5*fb, 0.5*fb]) + f_center_past
        f_band_future = np.array([-0.5*fb, 0.5*fb]) + f_center_future

        # Create frequency profiles
        f_past   = data.constrained_random_walk([f_band_past[0]] * (L * Np + time_shift), [f_band_past[1]] * (L * Np + time_shift), freq_change_rate)
        f_future = data.constrained_random_walk([f_band_future[0]] * (L * Nf - time_shift), [f_band_future[1]] * (L * Nf - time_shift), freq_change_rate, last_state=f_past[-1])
        frequency_profile = np.concatenate([f_past, f_future])

        # Create baseline wander profile
        baseline_wander_profile = data.smoothened_random_walk((Np + Nf) * L, baseline_wander_range[0], baseline_wander_range[1], fs, baseline_wander_fc, baseline_wander_change_rate)

        # Create amplitude profile
        amplitude_profile = data.smoothened_random_walk((Np + Nf) * L, amplitude_range[0], amplitude_range[1], fs, amplitude_fc, amplitude_change_rate)

        # Create noise profile
        noise_profile = data.create_noise_profile((Np + Nf) * L, noise_std)

        # Create final signal
        signal  = data.create_sine_profile(frequency_profile, fs)
        signal *= amplitude_profile
        signal += baseline_wander_profile
        signal += noise_profile

        # Reshape signal
        signal_reshaped = np.reshape(signal, (1, Np+Nf, L, 1))

        # Create time vector
        t = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)

        return t, signal, signal_reshaped

    def generate_signal_mp(task: Tuple[float, float, PPC.tools.Settings]) -> np.ndarray:
        """Wrapper around generate_signal with a single argument, to be used with multiprocessing.Pool

        Args:
            task (Tuple[float, float, PPC.tools.Settings]): past center frequency, future center frequency and generator settings

        Returns:
            np.ndarray: signal reshaped to segments of shape [1, Np+Nf, L, 1]
        """

        f_center_past, f_center_future, settings = task

        _, _, signal_reshaped = data.generate_signal(
            f_center_past               = f_center_past,
            f_center_future             = f_center_future,
            L                           = settings.L,
            time_shift_range            = settings.time_shift_range,
            noise_std_range             = settings.noise_std_range,
            fb                          = settings.fb,
            Np                          = settings.Np,
            Nf                          = settings.Nf,
            freq_change_rate            = settings.freq_change_rate,
            baseline_wander_range       = settings.baseline_wander_range,
            baseline_wander_fc          = settings.baseline_wander_fc,
            baseline_wander_change_rate = settings.baseline_wander_change_rate,
            amplitude_range             = settings.amplitude_range,
            amplitude_fc                = settings.amplitude_fc,
            amplitude_change_rate       = settings.amplitude_change_rate,
            fs                          = settings.fs,
        )

        return signal_reshaped

    class GridSampleHelper:
        """Picklable callable for `Pool.imap` that generates one sample for a given (past, future) center frequency pair"""

        def __init__(
            self,
            Np: int,
            Nf: int,
            L: int,
            fs: int,
            fr: Tuple[float, float],
            fb: float,
            freq_change_rate: float,
            baseline_wander_range: Tuple[float, float],
            baseline_wander_fc: float,
            baseline_wander_change_rate: float,
            amplitude_range: Tuple[float, float],
            amplitude_fc: float,
            amplitude_change_rate: float,
            noise_std_range: Tuple[float, float],
            time_shift_range: Tuple[float, float],
            seed: int,
        ):
            """Store the signal-generation parameters passed through to `data.generate_signal` on each call

            Args:
                fr (Tuple[float, float]): overall frequency range (unused directly, kept for interface parity)
                seed (int): seed of the dataset; sample i is generated with seed (seed, i)
                (all other args): see `data.generate_signal`
            """

            self.Np                          = Np
            self.Nf                          = Nf
            self.L                           = L
            self.fs                          = fs
            self.fr                          = fr
            self.fb                          = fb
            self.freq_change_rate            = freq_change_rate
            self.baseline_wander_range       = baseline_wander_range
            self.baseline_wander_fc          = baseline_wander_fc
            self.baseline_wander_change_rate = baseline_wander_change_rate
            self.amplitude_range             = amplitude_range
            self.amplitude_fc                = amplitude_fc
            self.amplitude_change_rate       = amplitude_change_rate
            self.noise_std_range             = noise_std_range
            self.time_shift_range            = time_shift_range
            self.seed                        = seed

        def __call__(self, task: Tuple[int, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Generate one sample for the given explicit (past, future) center-frequency pair

            Args:
                task (Tuple[int, Tuple[float, float]]): (sample index, (past center frequency, future center frequency)). The index seeds the generation of the sample.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: time vector, flat signal, and signal reshaped into (Np+Nf, L, 1) segments
            """

            index, (f_center_past, f_center_future) = task

            # Seed per sample, for reproducibility
            np.random.seed([self.seed, index])

            t, signal, signal_reshaped = data.generate_signal(
                f_center_past               = f_center_past,
                f_center_future             = f_center_future,
                L                           = self.L,
                time_shift_range            = self.time_shift_range,
                noise_std_range             = self.noise_std_range,
                fb                          = self.fb,
                Np                          = self.Np,
                Nf                          = self.Nf,
                freq_change_rate            = self.freq_change_rate,
                baseline_wander_range       = self.baseline_wander_range,
                baseline_wander_fc          = self.baseline_wander_fc,
                baseline_wander_change_rate = self.baseline_wander_change_rate,
                amplitude_range             = self.amplitude_range,
                amplitude_fc                = self.amplitude_fc,
                amplitude_change_rate       = self.amplitude_change_rate,
                fs                          = self.fs,
            )

            return t, signal, signal_reshaped[0]

    class ContrastiveSampleHelper:
        """Picklable callable for `Pool.imap` that generates one PCPC sample: a past signal, its positive continuation,
        and Nchi negative continuations in frequency bands at least fm Hz away
        """

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
            seed: int,
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
                seed (int): seed of the dataset; sample i is generated with seed (seed, i)
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
            self.seed                        = seed

        def __call__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Generate one past sample with its positive and negative future samples

            Args:
                index (int): index of the sample in the dataset, used to seed its generation

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: past sample [Np, L, 1], positive future sample [Nf, L, 1], negative future samples [Nchi, Nf, L, 1], positive frequency band [2] and negative frequency bands [Nchi, 2]
            """

            # Seed per sample, for reproducibility
            np.random.seed([self.seed, index])

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
            f_past       = data.constrained_random_walk([f_band_pos[0]] * (self.L * self.Np + time_shift), [f_band_pos[1]] * (self.L * self.Np + time_shift), self.freq_change_rate)
            f_future_pos = data.constrained_random_walk([f_band_pos[0]] * (self.L * self.Nf - time_shift), [f_band_pos[1]] * (self.L * self.Nf - time_shift), self.freq_change_rate, last_state=f_past[-1])
            f_future_neg = np.stack([data.constrained_random_walk([f_bands_neg[i, 0]] * (self.L * self.Nf - time_shift), [f_bands_neg[i, 1]] * (self.L * self.Nf - time_shift), self.freq_change_rate, last_state=f_past[-1]) for i in range(len(f_bands_neg))], axis=0)

            # Create baseline wander profile
            baseline_wander_profile = data.smoothened_random_walk((self.Np + self.Nf) * self.L, self.baseline_wander_range[0], self.baseline_wander_range[1], self.fs, self.baseline_wander_fc, self.baseline_wander_change_rate)

            # Create amplitude profile
            amplitude_profile = data.smoothened_random_walk((self.Np + self.Nf) * self.L, self.amplitude_range[0], self.amplitude_range[1], self.fs, self.amplitude_fc, self.amplitude_change_rate)

            # Create noise profile
            noise_profile = data.create_noise_profile((self.Np + self.Nf) * self.L, noise_std)

            # Create final signals
            sample_pos   = data.create_sine_profile(np.concatenate([f_past, f_future_pos]), self.fs)
            sample_pos  *= amplitude_profile
            sample_pos  += baseline_wander_profile
            sample_pos  += noise_profile

            samples_neg  = np.stack([data.create_sine_profile(np.concatenate([f_past, x]), self.fs) for x in f_future_neg], axis=0)
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

    class ContrastiveSamples:
        """Past, positive future and negative future samples (PCPC training and validation data), stored in an
        HDF5 file that is read batch by batch instead of loaded into memory
        """

        def __init__(self, path: Path):
            """Open a contrastive samples collection from an existing HDF5 file

            Args:
                path (Path): HDF5 file, as created by `ContrastiveSamples.create(...)`
            """

            self.handle = h5py.File(path, 'r')

            self.past_data       = self.handle['collection_past']       # [N, Np, L, 1]
            self.future_pos_data = self.handle['collection_future_pos'] # [N, Nf, L, 1]
            self.future_neg_data = self.handle['collection_future_neg'] # [N, Nchi, Nf, L, 1]

            self.Np   = self.past_data.shape[1]
            self.L    = self.past_data.shape[2]
            self.Nf   = self.future_pos_data.shape[1]
            self.Nchi = self.future_neg_data.shape[1]

        def create(
            path: Path,
            size: int,
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
            workers: int=1,
            seed: int=0,
        ) -> 'data.ContrastiveSamples':
            """Generate `size` contrastive samples and save them to an HDF5 file

            Args:
                path (Path): HDF5 file to write
                size (int): number of samples to generate
                workers (int, optional): number of worker processes to generate samples in parallel. Defaults to 1.
                seed (int, optional): seed of the dataset; sample i is generated with seed (seed, i). Defaults to 0.
                (all other args): see `data.ContrastiveSampleHelper`

            Returns:
                data.ContrastiveSamples: the generated collection, opened from the newly written file
            """

            if workers == 1:
                map_func   = map
                close_func = lambda: None
            elif workers > 1:
                pool       = Pool(workers)
                map_func   = pool.imap
                close_func = pool.close

            generator = data.ContrastiveSampleHelper(
                Np,
                Nf,
                L,
                Nchi,
                fs,
                fr,
                fb,
                fm,
                freq_change_rate,
                baseline_wander_range,
                baseline_wander_fc,
                baseline_wander_change_rate,
                amplitude_range,
                amplitude_fc,
                amplitude_change_rate,
                noise_std_range,
                time_shift_range,
                seed,
            )

            names = ['collection_past', 'collection_future_pos', 'collection_future_neg', 'collection_band_pos', 'collection_bands_neg']

            with h5py.File(path, 'w') as hf:

                # Write samples to the file as they are generated
                for i, sample in enumerate(tqdm(map_func(generator, range(size)), total=size)):
                    for name, x in zip(names, sample):
                        if i == 0:
                            hf.create_dataset(name, shape=(size,) + np.shape(x), dtype=np.asarray(x).dtype)
                        hf[name][i] = x

            close_func()

            return data.ContrastiveSamples(path)

        def close(self):
            """Close the HDF5 file"""

            self.handle.close()

    class ContrastiveSamplesDataset:
        """tf.data-backed, batched, shuffled generator over a `ContrastiveSamples` collection"""

        def __init__(self, ContrastiveSamples_inst: 'data.ContrastiveSamples', batch_size: int, seed: int=0):
            """
            Args:
                ContrastiveSamples_inst (data.ContrastiveSamples): opened contrastive samples collection to iterate over
                batch_size (int): batch size
                seed (int, optional): seed for shuffling the sample order. Defaults to 0.
            """

            self.ContrastiveSamples_inst = ContrastiveSamples_inst
            self.batch_size              = batch_size

            # Random number generator for shuffling
            self.rng = np.random.RandomState(seed)

            self.inds = np.arange(self.ContrastiveSamples_inst.past_data.shape[0])

        def generator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Endlessly yield shuffled batches of past, positive future and negative future segments

            Yields:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: shapes [B, Np, L, 1], [B, Nf, L, 1] and [B, Nchi, Nf, L, 1]
            """

            while True:
                self.rng.shuffle(self.inds)

                for i in range(int(len(self.inds)/self.batch_size)):

                    # HDF5 requires increasing indexes
                    inds = np.sort(self.inds[i*self.batch_size:(i+1)*self.batch_size])

                    past_data       = self.ContrastiveSamples_inst.past_data[inds, ...]
                    future_pos_data = self.ContrastiveSamples_inst.future_pos_data[inds, ...]
                    future_neg_data = self.ContrastiveSamples_inst.future_neg_data[inds, ...]

                    yield past_data, future_pos_data, future_neg_data

        def __iter__(self) -> Iterator:
            """Iterate over batches of the underlying tf.data.Dataset

            Returns:
                Iterator: iterator over (past, positive future, negative future) batches
            """

            return iter(self.tf_dataset())

        def tf_dataset(self) -> tf.data.Dataset:
            """Build the underlying, prefetched tf.data.Dataset

            Returns:
                tf.data.Dataset: dataset yielding (past, positive future, negative future) batches
            """

            Np, Nf, L, Nchi = self.ContrastiveSamples_inst.Np, self.ContrastiveSamples_inst.Nf, self.ContrastiveSamples_inst.L, self.ContrastiveSamples_inst.Nchi

            # Convert generator to TensorFlow dataset
            dataset = tf.data.Dataset.from_generator(
                generator        = self.generator,
                name             = 'generator',
                output_signature = (
                    tf.TensorSpec(shape=(self.batch_size, Np, L, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, Nf, L, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, Nchi, Nf, L, 1), dtype=tf.float32),
                ),
            )

            # Prefetch next batch for potentially better performance
            dataset = dataset.prefetch(
                buffer_size = tf.data.AUTOTUNE,
                name        = 'prefetch',
            )

            return dataset

    class GridSamples:
        """Samples on a grid of past/future center frequencies, for the test figures, stored in an HDF5 file"""

        def __init__(self, path: Path):
            """Load a grid-samples collection from an existing HDF5 file

            Args:
                path (Path): HDF5 file, as created by `GridSamples.create(...)`
            """

            with h5py.File(path, 'r') as hf:
                self.data                        = hf['collection'][:]
                self.gridsize                    = hf['collection'].attrs.get('gridsize')
                self.iterations                  = hf['collection'].attrs.get('iterations')
                self.Np                          = hf['collection'].attrs.get('Np')
                self.Nf                          = hf['collection'].attrs.get('Nf')
                self.L                           = hf['collection'].attrs.get('L')
                self.fs                          = hf['collection'].attrs.get('fs')
                self.fr                          = hf['collection'].attrs.get('fr')
                self.fb                          = hf['collection'].attrs.get('fb')
                self.freq_change_rate            = hf['collection'].attrs.get('freq_change_rate')
                self.baseline_wander_range       = hf['collection'].attrs.get('baseline_wander_range')
                self.baseline_wander_fc          = hf['collection'].attrs.get('baseline_wander_fc')
                self.baseline_wander_change_rate = hf['collection'].attrs.get('baseline_wander_change_rate')
                self.amplitude_range             = hf['collection'].attrs.get('amplitude_range')
                self.amplitude_fc                = hf['collection'].attrs.get('amplitude_fc')
                self.amplitude_change_rate       = hf['collection'].attrs.get('amplitude_change_rate')
                self.noise_std_range             = hf['collection'].attrs.get('noise_std_range')
                self.time_shift_range            = hf['collection'].attrs.get('time_shift_range')

            self.flat_data   = np.reshape(self.data, (self.data.shape[0], self.data.shape[1]*self.data.shape[2], self.data.shape[3]))
            self.past_data   = self.data[:, :self.Np, ...]
            self.future_data = self.data[:, self.Np:, ...]

            self.grid            = np.linspace(self.fr[0]+0.5*self.fb, self.fr[1]-0.5*self.fb, self.gridsize, endpoint=True)
            self.f_center_past   = np.repeat(np.repeat(self.grid, len(self.grid)), self.iterations)
            self.f_center_future = np.repeat(np.tile(self.grid, len(self.grid)), self.iterations)

        def create(
            path: Path,
            gridsize: int,
            iterations: int,
            Np: int,
            Nf: int,
            L: int,
            fs: int,
            fr: Tuple[float, float],
            fb: float,
            freq_change_rate: float,
            baseline_wander_range: Tuple[float, float],
            baseline_wander_fc: float,
            baseline_wander_change_rate: float,
            amplitude_range: Tuple[float, float],
            amplitude_fc: float,
            amplitude_change_rate: float,
            noise_std_range: Tuple[float, float],
            time_shift_range: Tuple[float, float],
            workers: int=1,
            seed: int=0,
        ) -> 'data.GridSamples':
            """Generate samples over a `gridsize` x `gridsize` grid of past/future center frequencies and save them

            `iterations` samples are generated for each of the `gridsize**2` frequency combinations.

            Args:
                path (Path): HDF5 file to write
                gridsize (int): number of distinct center frequencies per axis
                iterations (int): number of independent samples generated per grid point
                workers (int, optional): number of worker processes to generate samples in parallel. Defaults to 1.
                seed (int, optional): seed of the dataset; sample i is generated with seed (seed, i). Defaults to 0.
                (all other args): see `data.generate_signal`

            Returns:
                data.GridSamples: the generated collection, loaded from the newly written file
            """

            if workers == 1:
                map_func   = map
                close_func = lambda: None
            elif workers > 1:
                pool       = Pool(workers)
                map_func   = pool.imap
                close_func = pool.close

            generator = data.GridSampleHelper(
                Np,
                Nf,
                L,
                fs,
                fr,
                fb,
                freq_change_rate,
                baseline_wander_range,
                baseline_wander_fc,
                baseline_wander_change_rate,
                amplitude_range,
                amplitude_fc,
                amplitude_change_rate,
                noise_std_range,
                time_shift_range,
                seed,
            )

            grid        = np.linspace(fr[0]+0.5*fb, fr[1]-0.5*fb, gridsize, endpoint=True)
            grid_past   = np.repeat(grid, len(grid))
            grid_future = np.tile(grid, len(grid))
            grid_past   = np.repeat(grid_past, iterations)
            grid_future = np.repeat(grid_future, iterations)
            tasks       = np.stack([grid_past, grid_future], axis=1)

            with h5py.File(path, 'w') as hf:

                # Write samples to the file as they are generated
                for i, (_, _, signals_reshaped) in enumerate(tqdm(map_func(generator, enumerate(tasks)), total=len(tasks))):
                    if i == 0:
                        hf.create_dataset('collection', shape=(len(tasks),) + signals_reshaped.shape, dtype=signals_reshaped.dtype)
                    hf['collection'][i] = signals_reshaped

                close_func()

                hf['collection'].attrs.create('gridsize', data=gridsize)
                hf['collection'].attrs.create('iterations', data=iterations)
                hf['collection'].attrs.create('Np', data=Np)
                hf['collection'].attrs.create('Nf', data=Nf)
                hf['collection'].attrs.create('L', data=L)
                hf['collection'].attrs.create('fs', data=fs)
                hf['collection'].attrs.create('fr', data=fr)
                hf['collection'].attrs.create('fb', data=fb)
                hf['collection'].attrs.create('freq_change_rate', data= freq_change_rate)
                hf['collection'].attrs.create('baseline_wander_range', data=baseline_wander_range)
                hf['collection'].attrs.create('baseline_wander_fc', data=baseline_wander_fc)
                hf['collection'].attrs.create('baseline_wander_change_rate', data=baseline_wander_change_rate)
                hf['collection'].attrs.create('amplitude_range', data=amplitude_range)
                hf['collection'].attrs.create('amplitude_fc', data=amplitude_fc)
                hf['collection'].attrs.create('amplitude_change_rate', data=amplitude_change_rate)
                hf['collection'].attrs.create('noise_std_range', data=noise_std_range)
                hf['collection'].attrs.create('time_shift_range', data=time_shift_range)

            return data.GridSamples(path)

    class PositiveNegativeSamples:
        """Samples without (positive) and with (negative) frequency change, for the detection metrics, stored in an HDF5 file"""

        def __init__(self, path: Path):
            """Load a labelled positive/negative samples collection from an existing HDF5 file

            Args:
                path (Path): HDF5 file, as created by `PositiveNegativeSamples.create(...)`
            """

            with h5py.File(path, 'r') as hf:
                self.pos_samples = hf['pos_samples'][:]
                self.neg_samples = hf['neg_samples'][:]

            self.samples        = np.concatenate([self.pos_samples, self.neg_samples], axis=0)
            self.anomaly_labels = np.concatenate([np.zeros(len(self.pos_samples)), np.ones(len(self.neg_samples))])

        def create(
            path: Path,
            num_pos: int,
            Np: int,
            Nf: int,
            L: int,
            fs: int,
            fr: Tuple[float, float],
            fb: float,
            freq_change_rate: float,
            baseline_wander_range: Tuple[float, float],
            baseline_wander_fc: float,
            baseline_wander_change_rate: float,
            amplitude_range: Tuple[float, float],
            amplitude_fc: float,
            amplitude_change_rate: float,
            noise_std_range: Tuple[float, float],
            time_shift_range: Tuple[float, float],
            workers: int=1,
            seed: int=0,
        ) -> 'data.PositiveNegativeSamples':
            """Generate `num_pos` positive and `num_pos` negative samples and save them to an HDF5 file

            Negative samples have a different, randomly drawn center frequency after the change point.

            Args:
                path (Path): HDF5 file to write
                num_pos (int): number of positive (and, symmetrically, negative) samples to generate
                workers (int, optional): number of worker processes to generate samples in parallel. Defaults to 1.
                seed (int, optional): seed of the dataset. The frequencies are drawn with this seed, and sample i is generated with seed (seed, i), counting the negative samples after the positive ones. Defaults to 0.
                (all other args): see `data.generate_signal`

            Returns:
                data.PositiveNegativeSamples: the generated collection, loaded from the newly written file
            """

            if workers == 1:
                map_func   = map
                close_func = lambda: None
            elif workers > 1:
                pool       = Pool(workers)
                map_func   = pool.imap
                close_func = pool.close

            # Random number generator for drawing the center frequencies
            rng = np.random.RandomState(seed)

            generator = data.GridSampleHelper(
                Np,
                Nf,
                L,
                fs,
                fr,
                fb,
                freq_change_rate,
                baseline_wander_range,
                baseline_wander_fc,
                baseline_wander_change_rate,
                amplitude_range,
                amplitude_fc,
                amplitude_change_rate,
                noise_std_range,
                time_shift_range,
                seed,
            )

            # Generate positive samples
            fcs         = rng.uniform(fr[0]+0.5*fb, fr[1]-0.5*fb, num_pos)
            tasks       = np.stack([fcs, fcs], axis=1)

            # Write samples to the file as they are generated
            hf = h5py.File(path, 'w')

            for i, (_, _, signals_reshaped) in enumerate(tqdm(map_func(generator, enumerate(tasks)), total=len(tasks))):
                if i == 0:
                    hf.create_dataset('pos_samples', shape=(num_pos,) + signals_reshaped.shape, dtype=signals_reshaped.dtype)
                hf['pos_samples'][i] = signals_reshaped

            # Generate negative samples
            fcs_past   = rng.uniform(fr[0]+0.5*fb, fr[1]-0.5*fb, num_pos)
            fcs_future = np.empty_like(fcs_past)

            for i in range(len(fcs_past)):
                while True:
                    fcs_future_candidate = rng.uniform(fr[0]+0.5*fb, fr[1]-0.5*fb)
                    if fcs_future_candidate != fcs_past[i]:
                        fcs_future[i] = fcs_future_candidate
                        break

            tasks = np.stack([fcs_past, fcs_future], axis=1)

            for i, (_, _, signals_reshaped) in enumerate(tqdm(map_func(generator, enumerate(tasks, start=num_pos)), total=len(tasks))):
                if i == 0:
                    hf.create_dataset('neg_samples', shape=(num_pos,) + signals_reshaped.shape, dtype=signals_reshaped.dtype)
                hf['neg_samples'][i] = signals_reshaped

            close_func()
            hf.close()

            return data.PositiveNegativeSamples(path)

class colormaps:
    """Colormaps for the figures"""

    def get_modified_oranges():
        """Get the matplotlib 'Oranges' colormap with its lowest value forced to white

        Returns:
            matplotlib.colors.Colormap: modified 'Oranges' colormap
        """

        cmap = mpl.colormaps['Oranges']

        cmap._segmentdata['red'][0, 1:]   = 1
        cmap._segmentdata['green'][0, 1:] = 1
        cmap._segmentdata['blue'][0, 1:]  = 1

        return cmap
