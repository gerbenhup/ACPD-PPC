from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Tuple, Iterator, Optional, Union
import scipy.integrate
import scipy.signal
import h5py

# Parameters of the sine wave generator, shared by create_dataset.py (to generate data.h5) and test.py (to generate test signals)
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

        x = tf.keras.layers.Conv1D(filters*8, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Conv1D(filters*4, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Conv1D(filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(Ne, activation='linear')(x)

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

        sigma_min = float(1/(np.power(tf.float32.max, 1/Ne)*np.sqrt(2*np.pi)))

        inputs = tf.keras.Input((Ng,))

        x = tf.keras.layers.Dense(dense_units, activation='relu')(inputs)
        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='relu')(x)

        outputs_means = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Lambda(lambda y: PPC.math.exp(tf.math.maximum(y, sigma_min)), name='Exponential')(outputs_stds)

        outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='Stack1D')([outputs_means, outputs_stds])

        model = tf.keras.Model(inputs, outputs)

        return model

class data:
    """Sine wave signal generation"""

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

        The change point lies a random time shift (time_shift_range, as a fraction of L) after the end of the past segments.

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

class SineFrequencyDataset:
    """Dataset of past, positive future and negative future sine wave segments, read from the HDF5 file made by create_dataset.py"""

    def __init__(
        self,
        path: Path,
        dataset: str,
        split: np.ndarray,
        batch_size: int,
        driver: Optional[str]=None,
        seed: int=0
    ):
        """
        Args:
            path (Path): path to the HDF5 file
            dataset (str): which split to use, one of 'train', 'val', 'test'
            split (np.ndarray): relative sizes of the train, validation and test splits
            batch_size (int): batch size
            driver (Optional[str], optional): h5py file driver. Defaults to None.
            seed (int, optional): seed for the split. Defaults to 0.
        """

        # Input arguments
        self.path       = path
        self.dataset    = dataset
        self.split      = split
        self.batch_size = batch_size
        self.driver     = driver
        self.seed       = seed

        # Open HDF5 file
        self.handle = h5py.File(self.path, 'r', driver=self.driver)

        # Get parameters
        self.Np   = self.handle['collection_past'].shape[1]
        self.L    = self.handle['collection_past'].shape[2]
        self.Nf   = self.handle['collection_future_pos'].shape[1]
        self.Nchi = self.handle['collection_future_neg'].shape[1]

        # Get indexes belonging to dataset
        np.random.seed(self.seed)

        self.split  = self.split / np.sum(self.split)
        train_count = int(np.floor(self.split[0] * self.handle['collection_past'].shape[0]))
        val_count   = int(np.floor(self.split[1] * self.handle['collection_past'].shape[0]))
        test_count  = int(self.handle['collection_past'].shape[0] - train_count - val_count)

        indexes = np.arange(self.handle['collection_past'].shape[0], dtype=int)
        np.random.shuffle(indexes)

        train_indexes = indexes[:train_count]
        val_indexes   = indexes[train_count:-test_count]
        test_indexes  = indexes[-test_count:]

        if self.dataset == 'train':
            self.indexes = train_indexes
        elif self.dataset == 'val':
            self.indexes = val_indexes
        elif self.dataset == 'test':
            self.indexes = test_indexes
        else:
            raise ValueError('dataset {} unknown'.format(self.dataset))

        self.choice_ind = 0

    def choice(self) -> np.ndarray:
        """Draw the indexes of the next batch, reshuffling once all samples of the split have been used

        Returns:
            np.ndarray: indexes of the next batch
        """

        if self.choice_ind > len(self.indexes) - self.batch_size:
            np.random.shuffle(self.indexes)
            self.choice_ind = 0

        inds = self.indexes[self.choice_ind:self.choice_ind+self.batch_size]
        self.choice_ind += self.batch_size

        return inds

    def generator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Endlessly yield batches of past, positive future and negative future segments

        Yields:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: shapes [B, Np, L, 1], [B, Nf, L, 1] and [B, Nchi, Nf, L, 1]
        """

        while True:

            inds = self.choice()
            inds = np.sort(inds)

            batch_past       = self.handle['collection_past'][inds]
            batch_future_pos = self.handle['collection_future_pos'][inds]
            batch_future_neg = self.handle['collection_future_neg'][inds]

            yield batch_past, batch_future_pos, batch_future_neg

    def __iter__(self) -> Iterator:
        """Iterate over batches of the underlying tf.data.Dataset

        Returns:
            Iterator: iterator over (past, positive future, negative future) batches
        """

        return iter(self.tf_dataset())

    def tf_dataset(self) -> tf.data.Dataset:
        """Create TensorFlow dataset

        Returns:
            tf.data.Dataset: TensorFlow dataset
        """

        # Convert generator to TensorFlow dataset
        dataset = tf.data.Dataset.from_generator(
            generator        = self.generator,
            name             = 'generator',
            output_signature = (
                tf.TensorSpec(shape=(self.batch_size, self.Np, self.L, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, self.Nf, self.L, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, self.Nchi, self.Nf, self.L, 1), dtype=tf.float32),
            ),
        )

        # Prefetch next batch for potentially better performance
        dataset = dataset.prefetch(
            buffer_size = tf.data.AUTOTUNE,
            name        = 'prefetch',
        )

        return dataset

    def __del__(self):

        self.handle.close()

    def close(self):
        """Close the HDF5 file"""

        self.handle.close()