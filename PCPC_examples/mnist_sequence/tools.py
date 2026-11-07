from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Tuple, Iterator, Optional
import matplotlib as mpl

class models:
    """Encoder, autoregressive and forecast models (identical to the PPC MNIST example, without decoder)"""

    def encoder_model(Ne: int, filters: int, image_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create the encoder: two blocks of Conv2D (kernel size 3, ReLU), batch normalization and max pooling,
        with filters and 2*filters channels, and a linear dense layer, [B, H, W, 1] -> [B, Ne]

        Args:
            Ne (int): size of encoding vector
            filters (int): number of filters in the first convolutional layers
            image_shape (Tuple[int, int]): height and width of input image

        Returns:
            tf.keras.Model: TensorFlow encoder model
        """

        inputs = tf.keras.Input(image_shape + (1,))

        x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

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
        """Create a forecast model: three ReLU dense layers (dense_units, 2*dense_units, 4*dense_units units) and
        linear output layers for the means and log standard deviations, [B, Ng] -> [B, Ne, 2]

        Args:
            Ng (int): size of context vector
            Ne (int): size of encoding vector
            dense_units (int): number of dense units after first fully-connected layer

        Returns:
            tf.keras.Model: TensorFlow forecast model
        """

        inputs = tf.keras.Input((Ng,))

        x = tf.keras.layers.Dense(dense_units, activation='relu')(inputs)
        x = tf.keras.layers.Dense(dense_units*2, activation='relu')(x)
        x = tf.keras.layers.Dense(dense_units*4, activation='relu')(x)

        outputs_means = tf.keras.layers.Dense(Ne, activation='linear')(x)
        outputs_stds  = tf.keras.layers.Dense(Ne, activation='linear')(x)
        # Standard deviations, with clipped log standard deviations to avoid underflow and overflow
        outputs_stds  = tf.keras.layers.Lambda(lambda y: tf.math.exp(tf.clip_by_value(y, -20.0, 88.0)), name='Exponential')(outputs_stds)

        outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1), name='Stack1D')([outputs_means, outputs_stds])

        model = tf.keras.Model(inputs, outputs)

        return model

class augmentation:
    """Data augmentation"""

    def mnist_augmentation() -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """Create the past/positive future/negative future augmentation models used to perturb MNIST digit sequences during training

        Past samples are augmented less (rotation/zoom/translation factor 0.05) than future samples (factor 0.1).

        Returns:
            Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]: augmentation models for past, positive future and negative future samples
        """

        model_past = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.RandomRotation(
                    factor    = 0.05,
                    fill_mode = 'constant',
                ),
                tf.keras.layers.RandomZoom(
                    height_factor = (-0.05, 0.05),
                    width_factor  = (-0.05, 0.05),
                    fill_mode     = 'constant',
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor = (-0.05, 0.05),
                    width_factor  = (-0.05, 0.05),
                    fill_mode     = 'constant',
                )
            ])
        )

        model_future_pos = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.RandomRotation(
                    factor    = 0.1,
                    fill_mode = 'constant',
                ),
                tf.keras.layers.RandomZoom(
                    height_factor = (-0.1, 0.1),
                    width_factor  = (-0.1, 0.1),
                    fill_mode     = 'constant',
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor = (-0.1, 0.1),
                    width_factor  = (-0.1, 0.1),
                    fill_mode     = 'constant',
                )
            ])
        )

        model_future_neg = tf.keras.layers.TimeDistributed(
            tf.keras.layers.TimeDistributed(
                tf.keras.Sequential([
                    tf.keras.layers.RandomRotation(
                        factor    = 0.1,
                        fill_mode = 'constant',
                    ),
                    tf.keras.layers.RandomZoom(
                        height_factor = (-0.1, 0.1),
                        width_factor  = (-0.1, 0.1),
                        fill_mode     = 'constant',
                    ),
                    tf.keras.layers.RandomTranslation(
                        height_factor = (-0.1, 0.1),
                        width_factor  = (-0.1, 0.1),
                        fill_mode     = 'constant',
                    )
                ])
            )
        )

        return model_past, model_future_pos, model_future_neg

class data:
    """MNIST splits and successive-digit sequence datasets"""

    class MNIST:
        """MNIST split (train, validation or test), with the images indexed by digit label"""

        def __init__(self, dataset: str, seed: int=0):
            """Load an MNIST split and index it by digit label

            A fixed-seed stratified split of 1000 images is carved out of the Keras training set to
            serve as the validation set.

            Args:
                dataset (str): which split to load, one of 'train', 'val', 'test'
                seed (int, optional): seed for the train/validation split. Defaults to 0.
            """

            # Random number generator for the train/validation split
            rng = np.random.RandomState(seed)

            # Load data
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

            # Create validation dataset
            _, counts = np.unique(y_train, return_counts=True)
            val_counts = np.round(1000 * counts / np.sum(counts)).astype(int)

            val_inds = []

            for digit, count in zip(range(10), val_counts):
                inds = np.argwhere(y_train == digit).ravel()
                val_inds.append(rng.choice(inds, size=count, replace=False))

            val_inds = np.concatenate(val_inds)

            x_val = x_train[val_inds, :, :]
            y_val = y_train[val_inds]

            x_train = np.delete(x_train, val_inds, axis=0)
            y_train = np.delete(y_train, val_inds, axis=0)

            # Select dataset
            if dataset == 'train':
                x, y = x_train, y_train
            elif dataset == 'val':
                x, y = x_val, y_val
            elif dataset == 'test':
                x, y = x_test, y_test
            else:
                raise ValueError('Unknown dataset {}'.format(dataset))

            # Normalize images
            x = (x / 255).astype(np.float32)

            self.dataset_with_labels = [x, y]
            self.shape               = x.shape[1:]

            # Sort data
            unique_labels = np.unique(y)

            self.dataset_per_label = {}

            for unique_label in unique_labels:
                inds = np.argwhere(y == unique_label).ravel()
                self.dataset_per_label[unique_label] = x[inds, :, :]

    class MNISTSequenceDataset:
        """Dataset of successive-digit MNIST sequences, with Nchi negative future samples of other digits"""

        def __init__(
            self,
            MNIST_inst: object,
            batch_size: int,
            Np: int,
            Nf: int,
            Nchi: int,
            augmentation: Optional[tf.keras.Model]=None,
            seed: int=0,
        ):
            """Build a tf.data-backed generator of successive-digit sequences from an MNIST split

            Args:
                MNIST_inst (object): loaded MNIST split, as returned by data.MNIST
                batch_size (int): batch size
                Np (int): number of past samples per sequence
                Nf (int): number of future samples per sequence
                Nchi (int): number of negative samples per future sample
                augmentation (Optional[tf.keras.Model], optional): (past, positive future, negative future) augmentation models, as returned by augmentation.mnist_augmentation. Defaults to None.
                seed (int, optional): seed for drawing the sequences. Defaults to 0.
            """

            # Input arguments
            self.MNIST_inst   = MNIST_inst
            self.batch_size   = batch_size
            self.Np           = Np
            self.Nf           = Nf
            self.Nchi         = Nchi
            self.augmentation = augmentation

            self.image_shape = self.MNIST_inst.shape
            self.dataset     = self.MNIST_inst.dataset_per_label

            # Random number generator seed and number of passes over the dataset
            self.seed       = seed
            self.num_passes = 0

        def prepare_samples(self, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Sample one sequence of successive digits starting at a random digit, with negative samples of other digits

            Args:
                rng (np.random.RandomState): random number generator to draw with

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: past samples [Np, 28, 28, 1], positive future samples [Nf, 28, 28, 1] and negative future samples [Nchi, Nf, 28, 28, 1]
            """

            start         = rng.randint(0, 10)
            labels_past   = np.mod(np.arange(start, start + self.Np), 10)
            labels_future = np.mod(np.arange(start + self.Np, start + self.Np + self.Nf), 10)

            samples_past = []

            for label_past in labels_past:
                samples_past.append(self.dataset[label_past][rng.choice(self.dataset[label_past].shape[0])])

            samples_past = np.stack(samples_past, axis=0)
            samples_past = np.expand_dims(samples_past, axis=-1)

            samples_future_pos = []

            for label_future in labels_future:
                samples_future_pos.append(self.dataset[label_future][rng.choice(self.dataset[label_future].shape[0])])

            samples_future_pos = np.stack(samples_future_pos, axis=0)
            samples_future_pos = np.expand_dims(samples_future_pos, axis=-1)

            samples_future_neg = [] # [Nchi, Nf, 28, 28, 1]

            for i in range(self.Nchi):
                samples_future_neg.append([])
                for label_future in labels_future:
                    while True:
                        label_prop = rng.randint(0, 10)
                        if label_prop != label_future:
                            break
                    samples_future_neg[-1].append(self.dataset[label_prop][rng.choice(self.dataset[label_prop].shape[0])])
                samples_future_neg[-1] = np.stack(samples_future_neg[-1], axis=0)

            samples_future_neg = np.stack(samples_future_neg, axis=0)
            samples_future_neg = np.expand_dims(samples_future_neg, axis=-1)

            return samples_past, samples_future_pos, samples_future_neg

        def generator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Endlessly yield batches of past, positive future and negative future digit sequences

            Yields:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: batches of past, positive future and negative future samples
            """

            # New random number generator for every pass over the dataset
            rng = np.random.RandomState([self.seed, self.num_passes])
            self.num_passes += 1

            while True:

                # Allocate lists for samples
                batch_samples_past       = [None for _ in range(self.batch_size)]
                batch_samples_future_pos = [None for _ in range(self.batch_size)]
                batch_samples_future_neg = [None for _ in range(self.batch_size)]

                # Prepare samples with appropriate map function
                for i, (samples_past, samples_future_pos, samples_future_neg) in enumerate(map(self.prepare_samples, [rng]*self.batch_size)):
                    batch_samples_past[i], batch_samples_future_pos[i], batch_samples_future_neg[i] = samples_past, samples_future_pos, samples_future_neg

                # Stack samples in batches
                batch_samples_past       = np.stack(batch_samples_past, axis=0)
                batch_samples_future_pos = np.stack(batch_samples_future_pos, axis=0)
                batch_samples_future_neg = np.stack(batch_samples_future_neg, axis=0)

                # Yield samples
                yield batch_samples_past, batch_samples_future_pos, batch_samples_future_neg

        def __iter__(self) -> Iterator:
            """Iterate over batches of the underlying tf.data.Dataset

            Returns:
                Iterator: iterator over (past, positive future, negative future) batches
            """

            return iter(self.tf_dataset())

        def tf_dataset(self) -> tf.data.Dataset:
            """Build the underlying tf.data.Dataset, with augmentation and prefetching applied if configured

            Returns:
                tf.data.Dataset: dataset yielding (past, positive future, negative future) batches
            """

            dataset = tf.data.Dataset.from_generator(
                generator        = self.generator,
                name             = 'generator',
                output_signature = (
                    tf.TensorSpec(shape=(self.batch_size, self.Np) + self.image_shape + (1,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, self.Nf) + self.image_shape + (1,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, self.Nchi, self.Nf) + self.image_shape + (1,), dtype=tf.float32),
                )
            )

            if self.augmentation is not None:
                dataset = dataset.map(
                    map_func           = lambda x, y, z: [self.augmentation[0](x), self.augmentation[1](y), self.augmentation[2](z)],
                    num_parallel_calls = tf.data.AUTOTUNE,
                    deterministic      = True,
                )

            dataset = dataset.prefetch(
                buffer_size = tf.data.AUTOTUNE,
                name        = 'prefetch'
            )

            return dataset

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
