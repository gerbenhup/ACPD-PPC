from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Union, Optional, Tuple, Iterator, List
import scipy.signal
import scipy.integrate
from multiprocessing import Pool
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

class models:

    def encoder_model(Ne: int, filters: int, L: int) -> tf.keras.Model:
        """Create encoder model

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

    def decoder_model(L: int, Ne: int, filters: int) -> tf.keras.Model:
        """Create decoder model

        Args:
            L (int): length of signal segment
            Ne (int): size of encoding vector
            filters (int): number of filters in the corresponding encoder model

        Returns:
            tf.keras.Model: TensorFlow decoder model
        """
        inputs = tf.keras.Input((Ne,))

        x = tf.keras.layers.Dense(filters*2*(L//4), activation='relu')(inputs)
        x = tf.keras.layers.Reshape((L//4, filters*2))(x)

        x = tf.keras.layers.Conv1D(filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling1D()(x)

        x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling1D()(x)

        x = tf.keras.layers.Conv1D(1, kernel_size=3, activation='linear', padding='same')(x)

        model = tf.keras.Model(inputs, x)

        return model

    def ar_model(Np: int, Ne: int, Ng: int) -> tf.keras.Model:
        """Create autoregressive model

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
        """Create forecast model

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
        outputs_stds  = tf.keras.layers.Lambda(lambda y: PPC.math.exp(y), name='Exponential')(outputs_stds)

        outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='Stack1D')([outputs_means, outputs_stds])

        model = tf.keras.Model(inputs, outputs)

        return model

class data:

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
        Np,
        Nf,
        L,
        fs,
        fb,
        f_center_past,
        f_center_future,
        freq_change_rate,
        baseline_wander_range,
        baseline_wander_fc,
        baseline_wander_change_rate,
        amplitude_range,
        amplitude_fc,
        amplitude_change_rate,
        noise_std_range,
        time_shift_range,
    ):
        """Generate one (past, future) sine-wave segment with a frequency change point, baseline wander, amplitude modulation and noise

        Args:
            Np (int): number of past segments of length L
            Nf (int): number of future segments of length L
            L (int): segment length in samples
            fs (int): sampling frequency
            fb (float): width of the frequency band the center frequency is drawn from
            f_center_past (float): center frequency before the change point
            f_center_future (float): center frequency after the change point
            freq_change_rate (float): standard deviation of the frequency random walk
            baseline_wander_range (Tuple[float, float]): lower/upper bound of the baseline wander
            baseline_wander_fc (float): cutoff frequency of the baseline wander lowpass filter
            baseline_wander_change_rate (float): standard deviation of the baseline wander random walk
            amplitude_range (Tuple[float, float]): lower/upper bound of the amplitude modulation
            amplitude_fc (float): cutoff frequency of the amplitude modulation lowpass filter
            amplitude_change_rate (float): standard deviation of the amplitude modulation random walk
            noise_std_range (Tuple[float, float]): lower/upper bound of the additive noise standard deviation
            time_shift_range (Tuple[float, float]): fraction of a segment length by which the change point may be shifted

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: time vector, flat signal, and signal reshaped into (Np+Nf, L, 1) segments
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
        signal_reshaped = np.reshape(signal, (Np+Nf, L, 1))

        # Create time vector
        t = np.linspace(0, len(signal)/fs, len(signal), endpoint=False)

        return t, signal, signal_reshaped

    class positive_sample_helper:
        """Picklable callable that generates a single "positive" sample (no frequency change) for `Pool.imap`

        Stores the signal-generation parameters (see `data.generate_signal`) and, on each call, draws a
        random center frequency and generates a signal whose frequency stays the same across the change point.
        """

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
                fr (Tuple[float, float]): overall frequency range the center frequency is drawn from
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

        def __call__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Draw a random center frequency and generate one conforming, unchanged-frequency sample

            Args:
                index (int): index of the sample in the dataset, used to seed its generation

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: see `data.generate_signal`
            """

            # Seed per sample, so the dataset is reproducible regardless of which worker generates which sample
            np.random.seed([self.seed, index])

            f_center_past   = np.random.uniform(self.fr[0]+0.5*self.fb, self.fr[1]-0.5*self.fb)
            f_center_future = f_center_past

            return data.generate_signal(
                self.Np,
                self.Nf,
                self.L,
                self.fs,
                self.fb,
                f_center_past,
                f_center_future,
                self.freq_change_rate,
                self.baseline_wander_range,
                self.baseline_wander_fc,
                self.baseline_wander_change_rate,
                self.amplitude_range,
                self.amplitude_fc,
                self.amplitude_change_rate,
                self.noise_std_range,
                self.time_shift_range,
            )

    class grid_sample_helper:
        """Picklable callable that generates a single sample for a given (past, future) center-frequency pair

        Stores the signal-generation parameters (see `data.generate_signal`) and, on each call, generates a
        signal whose past/future center frequencies are given explicitly, allowing a systematic sweep over
        the frequency grid (used for both the grid-based test set and the labelled positive/negative sets).
        """

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
                Tuple[np.ndarray, np.ndarray, np.ndarray]: see `data.generate_signal`
            """

            index, (f_center_past, f_center_future) = task

            # Seed per sample, so the dataset is reproducible regardless of which worker generates which sample
            np.random.seed([self.seed, index])

            return data.generate_signal(
                self.Np,
                self.Nf,
                self.L,
                self.fs,
                self.fb,
                f_center_past,
                f_center_future,
                self.freq_change_rate,
                self.baseline_wander_range,
                self.baseline_wander_fc,
                self.baseline_wander_change_rate,
                self.amplitude_range,
                self.amplitude_fc,
                self.amplitude_change_rate,
                self.noise_std_range,
                self.time_shift_range,
            )

    class PositiveSamples:
        """Collection of "positive" signal segments (no frequency change across the segment boundary)

        Backed by an HDF5 file; used for training and validation, where every sample is a conforming
        (non-anomalous) sequence. Use `PositiveSamples.create(...)` to generate a new HDF5 file, or the
        constructor to load an existing one.
        """

        def __init__(self, path: Path):
            """Load a positive-samples collection from an existing HDF5 file

            Args:
                path (Path): HDF5 file, as created by `PositiveSamples.create(...)`
            """

            with h5py.File(path, 'r') as hf:
                self.data                        = hf['collection'][:]
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

        def create(
            path: Path,
            size: int,
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
        ) -> 'data.PositiveSamples':
            """Generate `size` positive samples and save them (and the generation parameters) to an HDF5 file

            Args:
                path (Path): HDF5 file to write
                size (int): number of samples to generate
                workers (int, optional): number of worker processes to generate samples in parallel. Defaults to 1.
                seed (int, optional): seed of the dataset; sample i is generated with seed (seed, i). Defaults to 0.
                (all other args): see `data.generate_signal`

            Returns:
                data.PositiveSamples: the generated collection, loaded from the newly written file
            """

            if workers == 1:
                map_func   = map
                close_func = lambda: None
            elif workers > 1:
                pool       = Pool(workers)
                map_func   = pool.imap
                close_func = pool.close

            generator = data.positive_sample_helper(
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

            collection = []
            for _, _, signals_reshaped in tqdm(map_func(generator, range(size)), total=size):
                collection.append(signals_reshaped)
            collection = np.stack(collection, axis=0)

            close_func()

            with h5py.File(path, 'w') as hf:
                hf.create_dataset('collection', data=collection)
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

            return data.PositiveSamples(path)

    class PositiveNegativeSamples:
        """Labelled collection of conforming ("positive") and frequency-changed ("negative") signal segments

        Backed by an HDF5 file; used for anomaly-detection metric evaluation (ROC/PRC, threshold search).
        Use `PositiveNegativeSamples.create(...)` to generate a new HDF5 file, or the constructor to load
        an existing one.
        """

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

            Positive samples share the same center frequency across the segment boundary; negative samples
            have a different (randomly redrawn) center frequency after the boundary.

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

            generator = data.grid_sample_helper(
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

            pos_samples = []
            for _, _, signals_reshaped in tqdm(map_func(generator, enumerate(tasks)), total=len(tasks)):
                pos_samples.append(signals_reshaped)
            pos_samples = np.stack(pos_samples, axis=0)

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

            neg_samples = []
            for _, _, signals_reshaped in tqdm(map_func(generator, enumerate(tasks, start=num_pos)), total=len(tasks)):
                neg_samples.append(signals_reshaped)
            neg_samples = np.stack(neg_samples, axis=0)

            close_func()

            with h5py.File(path, 'w') as hf:
                hf.create_dataset('pos_samples', data=pos_samples)
                hf.create_dataset('neg_samples', data=neg_samples)

            return data.PositiveNegativeSamples(path)

    class GridSamples:
        """Collection of signal segments generated over a systematic grid of past/future center frequencies

        Backed by an HDF5 file; used for the test-set log-likelihood/probability heatmap and example-signal
        figures. Use `GridSamples.create(...)` to generate a new HDF5 file, or the constructor to load an
        existing one.
        """

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

            For each of the `gridsize**2` (past, future) frequency combinations, `iterations` independent
            samples are generated (to average out noise/wander randomness at evaluation time).

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

            generator = data.grid_sample_helper(
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

            collection = []
            for _, _, signals_reshaped in tqdm(map_func(generator, enumerate(tasks)), total=len(tasks)):
                collection.append(signals_reshaped)
            collection = np.stack(collection, axis=0)

            close_func()

            with h5py.File(path, 'w') as hf:
                hf.create_dataset('collection', data=collection)
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

    class PositiveSamplesDataset:
        """tf.data-backed, batched, shuffled generator over a `PositiveSamples` collection"""

        def __init__(self, PositiveSample_inst: 'data.PositiveSamples', batch_size: int, seed: int=0):
            """
            Args:
                PositiveSample_inst (data.PositiveSamples): loaded positive-samples collection to iterate over
                batch_size (int): batch size
                seed (int, optional): seed of this dataset's own random number generator, used for shuffling sample order. Defaults to 0.
            """

            self.PositiveSample_inst = PositiveSample_inst
            self.batch_size          = batch_size

            # Own random number generator, so the sample order does not depend on other users of np.random
            self.rng = np.random.RandomState(seed)

            self.inds = np.arange(len(self.PositiveSample_inst.data))

        def generator(self) -> Tuple[np.ndarray, np.ndarray]:
            """Endlessly yield shuffled batches of (past, future) samples

            Yields:
                Tuple[np.ndarray, np.ndarray]: batch of past samples and batch of future samples
            """

            while True:
                self.rng.shuffle(self.inds)

                for i in range(int(len(self.inds)/self.batch_size)):

                    past_data   = self.PositiveSample_inst.past_data[i*self.batch_size:(i+1)*self.batch_size, ...]
                    future_data = self.PositiveSample_inst.future_data[i*self.batch_size:(i+1)*self.batch_size, ...]

                    yield past_data, future_data

        def __iter__(self) -> Iterator:
            """Iterate over batches of the underlying tf.data.Dataset

            Returns:
                Iterator: iterator over (past samples, future samples) batches
            """

            return iter(self.tf_dataset())

        def tf_dataset(self) -> tf.data.Dataset:
            """Build the underlying, prefetched tf.data.Dataset

            Returns:
                tf.data.Dataset: dataset yielding (past samples, future samples) batches
            """

            # Convert generator to TensorFlow dataset
            dataset = tf.data.Dataset.from_generator(
                generator        = self.generator,
                name             = 'generator',
                output_signature = (
                    tf.TensorSpec(shape=(self.batch_size, self.PositiveSample_inst.Np, self.PositiveSample_inst.L, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, self.PositiveSample_inst.Nf, self.PositiveSample_inst.L, 1), dtype=tf.float32),
                ),
            )

            # Prefetch next batch for potentially better performance
            dataset = dataset.prefetch(
                buffer_size = tf.data.AUTOTUNE,
                name        = 'prefetch',
            )

            return dataset

class callbacks:

    class SineMonitor:
        """Training-time callback that periodically plots train/validation losses"""

        def __init__(self, encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: List[tf.keras.Model], eval_iterations: int, fig_dir: Path):
            """
            Args:
                encoder_model (tf.keras.Model): encoder model
                decoder_model (tf.keras.Model): decoder model
                ar_model (tf.keras.Model): autoregressive model
                forecast_models (List[tf.keras.Model]): forecast models
                eval_iterations (int): number of training iterations between plot updates
                fig_dir (Path): directory to save the loss figure in
            """

            # Input parameters
            self.encoder_model           = encoder_model
            self.decoder_model           = decoder_model
            self.ar_model                = ar_model
            self.forecast_models         = forecast_models
            self.eval_iterations         = eval_iterations
            self.fig_dir                 = fig_dir

            # State parameters
            self.iterations   = 0
            self.train_losses = {}
            self.val_losses   = {}

        def __call__(self, **kwargs):
            """Record the losses reported for the current iteration and, periodically, save a loss figure

            Args:
                **kwargs: named loss values for the current iteration; names prefixed with 'val_' are recorded as validation losses
            """

            for loss_name, loss_value in kwargs.items():
                if loss_name[:4] == 'val_':
                    if loss_name[4:] not in self.val_losses:
                        self.val_losses[loss_name[4:]] = []
                    self.val_losses[loss_name[4:]].append(loss_value)
                else:
                    if loss_name not in self.train_losses:
                        self.train_losses[loss_name] = []
                    self.train_losses[loss_name].append(loss_value)

            self.iterations += 1

            if np.mod(self.iterations, self.eval_iterations) == 0:

                # Plot losses
                plt.figure(figsize=[10, 10])
                for i, loss_name in enumerate(self.train_losses):
                    plt.subplot(len(self.train_losses), 1, i+1)
                    plt.title(loss_name)
                    plt.plot(self.train_losses[loss_name], alpha=0.5, label='train')
                    plt.plot(self.val_losses[loss_name], alpha=0.5, label='val')
                    plt.grid()
                plt.legend()
                plt.tight_layout()
                try:
                    plt.savefig(self.fig_dir / 'losses.png')
                except Exception as e:
                    print(e)
                plt.close()

            return True

class colormaps:

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