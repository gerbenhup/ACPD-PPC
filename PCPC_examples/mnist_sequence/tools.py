from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Tuple, Iterator, Optional

class models:

    def encoder_model(Ne: int, filters: int, image_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create encoder model

        Args:
            Ne (int): size of encoding vector
            filters (int): number of filters in the first convolutional layers
            image_shape (Tuple[int, int]): height and width of input image

        Returns:
            tf.keras.Model: TensorFlow encoder model
        """

        inputs = tf.keras.Input(image_shape + (1,))

        x = tf.keras.layers.Conv2D(filters*2, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)

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

class augmentation:

    def MNISTaugmentation() -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """Create the past/positive future/negative future augmentation models used to perturb MNIST digit sequences during training

        Past samples are augmented more mildly (rotation/zoom/translation factor 0.05) than future
        samples (factor 0.1), so the model cannot trivially match past and future images pixel-for-pixel.

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

class MNISTSequenceDataset:
    """Dataset of successive-digit MNIST sequences, with Nchi negative future samples of other digits"""

    def __init__(
        self,
        dataset: str,
        batch_size: int,
        Np: int,
        Nf: int,
        Nchi: int,
        augmentation: Optional[tf.keras.Model]=None,
        seed: int=0
    ):
        """Load an MNIST split and index it by digit label

        A fixed-seed stratified split of 1000 images is carved out of the Keras training set to
        serve as the validation set.

        Args:
            dataset (str): which split to load, one of 'train', 'val', 'test'
            batch_size (int): batch size
            Np (int): number of past samples per sequence
            Nf (int): number of future samples per sequence
            Nchi (int): number of negative samples per future sample
            augmentation (Optional[tf.keras.Model], optional): (past, positive future, negative future) augmentation models, as returned by augmentation.MNISTaugmentation. Defaults to None.
            seed (int, optional): seed for the train/validation split. Defaults to 0.
        """

        # Input arguments
        self.batch_size   = batch_size
        self.Np           = Np
        self.Nf           = Nf
        self.Nchi         = Nchi
        self.augmentation = augmentation

        # Set seed
        np.random.seed(seed)

        # Load data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Create validation dataset
        _, counts = np.unique(y_train, return_counts=True)
        val_counts = np.round(1000 * counts / np.sum(counts)).astype(int)

        val_inds = []

        for digit, count in zip(range(10), val_counts):
            inds = np.argwhere(y_train == digit).ravel()
            val_inds.append(np.random.choice(inds, size=count, replace=False))

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

        self.image_shape = x_train.shape[1:]

        # Normalize images
        x = (x / 255).astype(np.float32)

        # Sort data
        unique_labels = np.unique(y)

        self.dataset = {}

        for unique_label in unique_labels:
            inds = np.argwhere(y == unique_label).ravel()
            self.dataset[unique_label] = x[inds, :, :]

    def prepare_samples(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample one sequence of successive digits starting at a random digit, with negative samples of other digits

        Args:
            *args, **kwargs: unused, present so this method can be passed to `map()` over a dummy iterable

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: past samples [Np, 28, 28, 1], positive future samples [Nf, 28, 28, 1] and negative future samples [Nchi, Nf, 28, 28, 1]
        """

        start         = np.random.randint(0, 10)
        labels_past   = np.mod(np.arange(start, start + self.Np), 10)
        labels_future = np.mod(np.arange(start + self.Np, start + self.Np + self.Nf), 10)

        samples_past = []

        for label_past in labels_past:
            samples_past.append(self.dataset[label_past][np.random.choice(self.dataset[label_past].shape[0])])

        samples_past = np.stack(samples_past, axis=0)
        samples_past = np.expand_dims(samples_past, axis=-1)

        samples_future_pos = []

        for label_future in labels_future:
            samples_future_pos.append(self.dataset[label_future][np.random.choice(self.dataset[label_future].shape[0])])

        samples_future_pos = np.stack(samples_future_pos, axis=0)
        samples_future_pos = np.expand_dims(samples_future_pos, axis=-1)

        samples_future_neg = [] # [Nchi, Nf, 28, 28, 1]

        for i in range(self.Nchi):
            samples_future_neg.append([])
            for label_future in labels_future:
                while True:
                    label_prop = np.random.randint(0, 10)
                    if label_prop != label_future:
                        break
                samples_future_neg[-1].append(self.dataset[label_prop][np.random.choice(self.dataset[label_prop].shape[0])])
            samples_future_neg[-1] = np.stack(samples_future_neg[-1], axis=0)

        samples_future_neg = np.stack(samples_future_neg, axis=0)
        samples_future_neg = np.expand_dims(samples_future_neg, axis=-1)

        return samples_past, samples_future_pos, samples_future_neg

    def generator(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Endlessly yield batches of past, positive future and negative future digit sequences

        Yields:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: batches of past, positive future and negative future samples
        """

        while True:

            # Allocate lists for samples
            batch_samples_past       = [None for _ in range(self.batch_size)]
            batch_samples_future_pos = [None for _ in range(self.batch_size)]
            batch_samples_future_neg = [None for _ in range(self.batch_size)]

            # Prepare samples with appropriate map function
            for i, (samples_past, samples_future_pos, samples_future_neg) in enumerate(map(self.prepare_samples, [None]*self.batch_size)):
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
                deterministic      = False,
            )

        dataset = dataset.prefetch(
            buffer_size = tf.data.AUTOTUNE,
            name        = 'prefetch'
        )

        return dataset