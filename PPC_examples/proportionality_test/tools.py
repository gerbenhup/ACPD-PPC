from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
from typing import Tuple, Iterator, Optional, List
import numpy as np
import tensorflow as tf
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

class models:
    """Encoder, decoder, autoregressive and forecast models"""

    def encoder_model(Ne: int) -> tf.keras.Model:
        """Create the encoder: one linear dense layer, [B, 1] -> [B, Ne]

        Args:
            Ne (int): size of encoding vector

        Returns:
            tf.keras.Model: TensorFlow encoder model
        """

        inputs  = tf.keras.Input((1,))
        outputs = tf.keras.layers.Dense(Ne, activation='linear')(inputs)

        model = tf.keras.Model(inputs, outputs)

        return model

    def decoder_model(Ne: int) -> tf.keras.Model:
        """Create the decoder: one linear dense layer, [B, Ne] -> [B, 1]

        Args:
            Ne (int): size of encoding vector

        Returns:
            tf.keras.Model: TensorFlow decoder model
        """
        inputs  = tf.keras.Input((Ne,))
        outputs = tf.keras.layers.Dense(1, activation='linear')(inputs)

        model = tf.keras.Model(inputs, outputs)

        return model

    def ar_model(Ne: int, Ng: int) -> tf.keras.Model:
        """Create the autoregressive model: a GRU with Ng units, [B, 1, Ne] -> [B, 1, Ng]

        Args:
            Ne (int): size of encoding vector
            Ng (int): size of context vector

        Returns:
            tf.keras.Model: TensorFlow autoregressive model
        """

        inputs  = tf.keras.Input((1, Ne))
        outputs = tf.keras.layers.GRU(Ng, return_sequences=True, unroll=True)(inputs)
        model   = tf.keras.Model(inputs, outputs)

        return model

    def forecast_model(Ng: int, Ne: int, dense_units: int) -> tf.keras.Model:
        """Create a forecast model: six swish dense layers (dense_units, dense_units, 2*dense_units, 2*dense_units,
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

        x = tf.keras.layers.Dense(dense_units, activation='swish')(inputs)
        x = tf.keras.layers.Dense(dense_units, activation='swish')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='swish')(x)
        x = tf.keras.layers.Dense(dense_units*2, activation='swish')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='swish')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='swish')(x)

        outputs_means = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Lambda(lambda y: PPC.math.exp(y), name='Exponential')(outputs_stds)

        outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='Stack1D')([outputs_means, outputs_stds])

        model = tf.keras.Model(inputs, outputs)

        return model

class data:
    """Two-step generative process (x1, x2) and its tf.data dataset"""

    class TwoStepDataset:
        """tf.data-backed generator of (x1, x2) sample pairs from a two-step generative process"""

        def __init__(self, batch_size: int, step1_generator: callable, step2_generator: callable, seed: int=0):
            """
            Args:
                batch_size (int): batch size
                step1_generator (callable): callable(batch_size, rng) -> x1 samples
                step2_generator (callable): callable(x1 samples, rng) -> x2 samples
                seed (int, optional): seed of this dataset's own random number generator, which is passed to the step generators. Defaults to 0.
            """

            self.batch_size      = batch_size
            self.step1_generator = step1_generator
            self.step2_generator = step2_generator

            # Own random number generator, so the drawn samples do not depend on other users of np.random
            self.rng = np.random.RandomState(seed)

        def prepare_batch(self) -> Tuple[np.ndarray, np.ndarray]:
            """Draw one batch of (x1, x2) sample pairs

            Returns:
                Tuple[np.ndarray, np.ndarray]: batch of x1 samples and batch of x2 samples
            """

            samples_past   = self.step1_generator(self.batch_size, self.rng) # [B,]
            samples_future = self.step2_generator(samples_past, self.rng) # [B,]

            samples_past   = np.reshape(samples_past, (self.batch_size, 1, 1))
            samples_future = np.reshape(samples_future, (self.batch_size, 1, 1))

            return samples_past, samples_future

        def generator(self) -> Tuple[np.ndarray, np.ndarray]:
            """Endlessly yield batches of (x1, x2) sample pairs

            Yields:
                Tuple[np.ndarray, np.ndarray]: batch of x1 samples and batch of x2 samples
            """

            while True:
                yield self.prepare_batch()

        def __iter__(self) -> Iterator:
            """Iterate over batches of the underlying tf.data.Dataset

            Returns:
                Iterator: iterator over (x1 samples, x2 samples) batches
            """

            return iter(self.tf_dataset())

        def tf_dataset(self) -> tf.data.Dataset:
            """Build the underlying, prefetched tf.data.Dataset

            Returns:
                tf.data.Dataset: dataset yielding (x1 samples, x2 samples) batches
            """

            dataset = tf.data.Dataset.from_generator(
                generator        = self.generator,
                name             = 'generator',
                output_signature = (
                    tf.TensorSpec(shape=(self.batch_size, 1, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 1, 1), dtype=tf.float32),
                )
            )

            dataset = dataset.prefetch(
                buffer_size = tf.data.AUTOTUNE,
                name        = 'prefetch'
            )

            return dataset

    class ChoiceStepOne:
        """Step-1 generator that draws x1 from a fixed, discrete set of values"""

        def __init__(self, values: np.ndarray, probs: Optional[np.ndarray]=None):
            """
            Args:
                values (np.ndarray): discrete set of values to draw x1 from
                probs (Optional[np.ndarray], optional): sampling probability for each value. Defaults to None (uniform).
            """

            self.values = values
            self.probs  = probs

        def __call__(self, batch_size: int, rng: np.random.RandomState) -> np.ndarray:
            """Draw a batch of x1 samples

            Args:
                batch_size (int): number of samples to draw
                rng (np.random.RandomState): random number generator to draw with

            Returns:
                np.ndarray: x1 samples
            """

            return rng.choice(self.values, size=batch_size, p=self.probs)

    class NormalStepTwo:
        """Step-2 generator that draws x2 ~ Normal(x1, std_func(x1)) given x1 samples"""

        def __init__(self, std_func: callable):
            """
            Args:
                std_func (callable): callable(x1 samples) -> standard deviation of x2 for each sample
            """

            self.std_func = std_func

        def __call__(self, samples_future: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
            """Draw x2 samples conditioned on x1 samples

            Args:
                samples_future (np.ndarray): x1 samples to condition on
                rng (np.random.RandomState): random number generator to draw with

            Returns:
                np.ndarray: x2 samples
            """

            return rng.normal(samples_future, self.std_func(samples_future)).astype(np.float32)

class functions:
    """Gaussian density and its fit to an estimated density"""

    def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Gaussian probability density function

        Args:
            x (np.ndarray): points to evaluate the density at
            mu (float): mean
            sigma (float): standard deviation

        Returns:
            np.ndarray: density values
        """
        return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(1/2)*np.square((x-mu)/sigma))

    def fit_gaussian(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Fit a Gaussian density to (x, y) samples via nonlinear least squares

        Args:
            x (np.ndarray): sample points
            y (np.ndarray): density values at the sample points

        Returns:
            Tuple[float, float, float]: fitted mean, fitted standard deviation, mean absolute error of the fit
        """

        mu, sigma = scipy.optimize.curve_fit(
            f     = functions.gaussian,
            xdata = x,
            ydata = y,
        )[0]

        mae = np.mean(np.abs(functions.gaussian(x, mu, sigma) - y))

        return mu, sigma, mae

class callbacks:
    """Training-time callbacks"""

    class GaussianParameterMonitor:
        """Training-time callback that periodically fits and plots the estimated vs. true Gaussian parameters

        For each x1 value in `choice_step_one_values`, tracks how well the estimated x2 distribution's
        (mu, sigma) match the ground truth `x1, normal_a*x1+normal_b`, alongside the training losses.
        """

        def __init__(self, encoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: List[tf.keras.Model], choice_step_one_values: List[float], normal_a: float, normal_b: float, grid_res: float, eval_iterations: int, fig_path: Path):
            """
            Args:
                encoder_model (tf.keras.Model): encoder model
                ar_model (tf.keras.Model): autoregressive model
                forecast_models (List[tf.keras.Model]): forecast models
                choice_step_one_values (List[float]): x1 values to track
                normal_a (float): slope relating x1 to the true x2 standard deviation
                normal_b (float): intercept relating x1 to the true x2 standard deviation
                grid_res (float): resolution of the x2 evaluation grid
                eval_iterations (int): number of training iterations between evaluations
                fig_path (Path): path to save the diagnostic figure to
            """

            # Input parameters
            self.encoder_model          = encoder_model
            self.ar_model               = ar_model
            self.forecast_models        = forecast_models
            self.choice_step_one_values = choice_step_one_values
            self.normal_a               = normal_a
            self.normal_b               = normal_b
            self.grid_res               = grid_res
            self.eval_iterations        = eval_iterations
            self.fig_path               = fig_path

            # Stacked/distributed models
            self.encoder_model_t1d = PPC.wrappers.time_distributed(self.encoder_model)
            self.forecast_model_ts = PPC.wrappers.time_stacked(self.forecast_models, [False])

            # Create grid
            grid_min  = np.min(self.choice_step_one_values) - 6*(self.normal_a*np.min(self.choice_step_one_values)+self.normal_b)
            grid_max  = np.max(self.choice_step_one_values) + 6*(self.normal_a*np.max(self.choice_step_one_values)+self.normal_b)
            self.grid = np.arange(grid_min, grid_max+self.grid_res, self.grid_res)

            # Create samples
            self.samples_past   = [np.reshape(np.repeat(choice_step_one_value, len(self.grid)), (len(self.grid), 1, 1)) for choice_step_one_value in self.choice_step_one_values]
            self.samples_future = np.reshape(self.grid, (len(self.grid), 1, 1))

            # Determine true parameters
            self.parameters_true = np.array([[choice_step_one_value, self.normal_a*choice_step_one_value+self.normal_b] for choice_step_one_value in self.choice_step_one_values])

            # Create counter and tracker
            self.iterations      = 0
            self.parameters_pred = []
            self.forecast_loss   = []
            self.rec_loss        = []

        def __call__(self, **kwargs):
            """Record the losses reported for the current iteration and, periodically, save a diagnostic figure

            Args:
                **kwargs: named loss values for the current iteration, must include 'forecast_loss' and 'reconstruction_loss'
            """

            self.forecast_loss.append(kwargs['forecast_loss'])
            self.rec_loss.append(kwargs['reconstruction_loss'])

            self.iterations += 1

            if np.mod(self.iterations, self.eval_iterations) == 0:

                self.parameters_pred.append([])

                for i in range(len(self.choice_step_one_values)):

                    encodings_past        = self.encoder_model_t1d(self.samples_past[i])
                    encodings_future_pred = self.forecast_model_ts(self.ar_model(encodings_past)[:, -1, :])
                    encodings_future_true = self.encoder_model_t1d(self.samples_future)

                    EDF_pred  = np.array(PPC.probability.gaussian_exp_norm_log_likelihood(encodings_future_true, encodings_future_pred)[:, 0])
                    EDF_pred /= scipy.integrate.trapezoid(EDF_pred, self.grid)

                    try:
                        mu, sigma = functions.fit_gaussian(self.grid, EDF_pred)[:2]
                        self.parameters_pred[-1].append([mu, sigma])
                    except RuntimeError:
                        self.parameters_pred[-1].append([np.nan, np.nan])

                pars_pred  = np.array(self.parameters_pred)
                iterations = np.arange(len(pars_pred))*self.eval_iterations

                # Get pyplot color cycle
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                plt.figure(figsize=[10, 10])
                ax = plt.subplot(4, 1, 1)
                plt.title('mu')
                for i in range(len(self.choice_step_one_values)):
                    plt.plot(iterations, pars_pred[:, i, 0], color=colors[i])
                    plt.axhline(self.parameters_true[i, 0], color=colors[i], linestyle='--')
                plt.grid()

                plt.subplot(4, 1, 2, sharex=ax)
                plt.title('sigma')
                for i in range(len(self.choice_step_one_values)):
                    plt.plot(iterations, pars_pred[:, i, 1], color=colors[i])
                    plt.axhline(self.parameters_true[i, 1], color=colors[i], linestyle='--')
                plt.grid()

                plt.subplot(4, 1, 3, sharex=ax)
                plt.title('forecast_loss')
                plt.plot(self.forecast_loss)
                plt.grid()

                plt.subplot(4, 1, 4, sharex=ax)
                plt.title('rec_loss')
                plt.plot(self.rec_loss)
                plt.grid()

                plt.tight_layout()
                try:
                    plt.savefig(self.fig_path)
                except Exception as e:
                    print(e)
                plt.close()

            return True