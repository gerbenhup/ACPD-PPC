from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tensorflow as tf
import numpy as np
from typing import Tuple, Iterator, Optional, List
import matplotlib.pyplot as plt
import matplotlib as mpl

class models:
    """Encoder, decoder, autoregressive and forecast models"""

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

    def decoder_model(Ne: int, filters: int, image_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create the decoder, mirroring the encoder: a linear dense layer reshaped to [H/4, W/4, 2*filters], two blocks
        of Conv2D (kernel size 3, ReLU) and upsampling, and a linear Conv2D output layer, [B, Ne] -> [B, H, W, 1]

        Args:
            Ne (int): size of encoding vector
            filters (int): number of filters in the corresponding encoder model
            image_shape (Tuple[int, int]): input shape of the corresponding encoder model

        Returns:
            tf.keras.Model: TensorFlow decoder model
        """
        inputs = tf.keras.Input((Ne,))

        x = tf.keras.layers.Dense(filters*2*(image_shape[0]//4)*(image_shape[1]//4))(inputs)
        x = tf.keras.layers.Reshape((image_shape[0]//4, image_shape[1]//4, filters*2))(x)

        x = tf.keras.layers.Conv2D(filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.Conv2D(1, kernel_size=3, activation='linear', padding='same')(x)

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

    def mnist_augmentation() -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Create the past/future augmentation models used to perturb MNIST digit sequences during training

        Past samples are augmented less (rotation/zoom/translation factor 0.05) than future samples (factor 0.1).

        Returns:
            Tuple[tf.keras.Model, tf.keras.Model]: augmentation models for past and future samples
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

        model_future = tf.keras.layers.TimeDistributed(
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

        return model_past, model_future

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
        """Dataset of successive-digit MNIST sequences"""

        def __init__(
            self,
            MNIST_inst: object,
            batch_size: int,
            Np: int,
            Nf: int,
            augmentation: Optional[tf.keras.Model]=None,
            seed: int=0,
        ):
            """Build a tf.data-backed generator of successive-digit sequences from an MNIST split

            Args:
                MNIST_inst (object): loaded MNIST split, as returned by data.MNIST
                batch_size (int): batch size
                Np (int): number of past samples per sequence
                Nf (int): number of future samples per sequence
                augmentation (Optional[tf.keras.Model], optional): (past, future) augmentation models, as returned by augmentation.mnist_augmentation. Defaults to None.
                seed (int, optional): seed for drawing the sequences. Defaults to 0.
            """

            # Input arguments
            self.MNIST_inst   = MNIST_inst
            self.batch_size   = batch_size
            self.Np           = Np
            self.Nf           = Nf
            self.augmentation = augmentation

            # Random number generator seed and number of passes over the dataset
            self.seed       = seed
            self.num_passes = 0

            self.image_shape = self.MNIST_inst.shape
            self.dataset     = self.MNIST_inst.dataset_per_label

        def prepare_samples(self, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
            """Sample one past/future sequence of successive digits starting at a random digit

            Args:
                rng (np.random.RandomState): random number generator to draw with

            Returns:
                Tuple[np.ndarray, np.ndarray]: past samples of shape (Np, ...) and future samples of shape (Nf, ...)
            """

            start         = rng.randint(0, 10)
            labels_past   = np.mod(np.arange(start, start + self.Np), 10)
            labels_future = np.mod(np.arange(start + self.Np, start + self.Np + self.Nf), 10)

            samples_past = []

            for label_past in labels_past:
                samples_past.append(self.dataset[label_past][rng.choice(self.dataset[label_past].shape[0])])

            samples_past = np.stack(samples_past, axis=0)
            samples_past = np.expand_dims(samples_past, axis=-1)

            samples_future = []

            for label_future in labels_future:
                samples_future.append(self.dataset[label_future][rng.choice(self.dataset[label_future].shape[0])])

            samples_future = np.stack(samples_future, axis=0)
            samples_future = np.expand_dims(samples_future, axis=-1)

            return samples_past, samples_future

        def generator(self) -> Tuple[np.ndarray, np.ndarray]:
            """Endlessly yield batches of past/future digit sequences

            Yields:
                Tuple[np.ndarray, np.ndarray]: batch of past samples and batch of future samples
            """

            # New random number generator for every pass over the dataset
            rng = np.random.RandomState([self.seed, self.num_passes])
            self.num_passes += 1

            while True:

                # Allocate lists for samples
                batch_samples_past   = [None for _ in range(self.batch_size)]
                batch_samples_future = [None for _ in range(self.batch_size)]

                # Prepare samples with appropriate map function
                for i, (samples_past, samples_future) in enumerate(map(self.prepare_samples, [rng]*self.batch_size)):
                    batch_samples_past[i], batch_samples_future[i], = samples_past, samples_future

                # Stack samples in batches
                batch_samples_past   = np.stack(batch_samples_past, axis=0)
                batch_samples_future = np.stack(batch_samples_future, axis=0)

                # Yield samples
                yield batch_samples_past, batch_samples_future

        def __iter__(self) -> Iterator:
            """Iterate over batches of the underlying tf.data.Dataset

            Returns:
                Iterator: iterator over (past samples, future samples) batches
            """

            return iter(self.tf_dataset())

        def tf_dataset(self) -> tf.data.Dataset:
            """Build the underlying tf.data.Dataset, with augmentation and prefetching applied if configured

            Returns:
                tf.data.Dataset: dataset yielding (past samples, future samples) batches
            """

            dataset = tf.data.Dataset.from_generator(
                generator        = self.generator,
                name             = 'generator',
                output_signature = (
                    tf.TensorSpec(shape=(self.batch_size, self.Np) + self.image_shape + (1,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, self.Nf) + self.image_shape + (1,), dtype=tf.float32),
                )
            )

            if self.augmentation is not None:
                dataset = dataset.map(
                    map_func           = lambda x, y: [self.augmentation[0](x), self.augmentation[1](y)],
                    num_parallel_calls = tf.data.AUTOTUNE,
                    deterministic      = True,
                )

            dataset = dataset.prefetch(
                buffer_size = tf.data.AUTOTUNE,
                name        = 'prefetch'
            )

            return dataset

class callbacks:
    """Training-time callbacks"""

    class MNISTMonitor:
        """Training-time callback that periodically plots losses, reconstructions, latent space and prediction matrix"""

        def __init__(self, encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: List[tf.keras.Model], MNIST_inst: data.MNIST, eval_examples_per_digit: int, eval_iterations: int, fig_dir: Path):
            """
            Args:
                encoder_model (tf.keras.Model): encoder model
                decoder_model (tf.keras.Model): decoder model
                ar_model (tf.keras.Model): autoregressive model
                forecast_models (List[tf.keras.Model]): forecast models
                MNIST_inst (data.MNIST): MNIST split used to draw evaluation examples from
                eval_examples_per_digit (int): number of examples per digit used for evaluation plots
                eval_iterations (int): number of training iterations between evaluations
                fig_dir (Path): directory to save the diagnostic figures in
            """

            # Input parameters
            self.encoder_model           = encoder_model
            self.decoder_model           = decoder_model
            self.ar_model                = ar_model
            self.forecast_models         = forecast_models
            self.MNIST_inst              = MNIST_inst
            self.eval_examples_per_digit = eval_examples_per_digit
            self.eval_iterations         = eval_iterations
            self.fig_dir                 = fig_dir

            # State parameters
            self.iterations   = 0
            self.train_losses = {}
            self.val_losses   = {}

        def __call__(self, **kwargs):
            """Record the losses reported for the current iteration and, periodically, save diagnostic figures

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

                # Fetch examples
                example_inds = []
                for i in range(10):
                    example_inds.append(np.random.choice(np.argwhere(self.MNIST_inst.dataset_with_labels[1] == i).ravel(), size=self.eval_examples_per_digit, replace=False))
                example_inds = np.concatenate(example_inds)

                example_imgs   = np.expand_dims(self.MNIST_inst.dataset_with_labels[0][example_inds], axis=-1)
                example_labels = self.MNIST_inst.dataset_with_labels[1][example_inds]

                # Fetch model outputs
                encodings             = self.encoder_model(example_imgs)
                contexts              = self.ar_model.predict(tf.expand_dims(encodings, axis=1))[:, 0, :]
                encodings_future_pred = self.forecast_models[0].predict(contexts)
                decodings             = self.decoder_model(encodings)

                # Plot reconstruction examples
                plt.figure(figsize=[10, 10])
                for i in range(10):
                    ind = int(np.argwhere(example_labels == i).ravel()[0])
                    plt.subplot(5, 4, i*2+1)
                    plt.imshow(example_imgs[ind, :, :, 0], vmin=0, vmax=1)
                    plt.subplot(5, 4, i*2+2)
                    plt.imshow(decodings[ind, :, :, 0], vmin=0, vmax=1)
                try:
                    plt.savefig(self.fig_dir / 'reconstructions.png')
                except Exception as e:
                    print(e)
                plt.close()

                # Plot latent space example
                reduced_encodings = PPC.dimension.PCA(2).fit_transform(encodings)

                plt.figure(figsize=[10, 10])
                for i in range(10):
                    inds = np.argwhere(example_labels == i).ravel()
                    plt.scatter(reduced_encodings[inds, 0], reduced_encodings[inds, 1], marker='.', label='{}'.format(i))
                plt.legend()
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                try:
                    plt.savefig(self.fig_dir / 'latent_space.png')
                except Exception as e:
                    print(e)
                plt.close()

                # Plot prediction matrix
                pred_matrix = np.zeros((10, 10), dtype=np.float32)

                for ind_a in range(len(example_imgs)):
                    for ind_b in range(len(example_imgs)):
                        pred_matrix[example_labels[ind_a], example_labels[ind_b]] += PPC.probability.gaussian_log_likelihood(encodings[ind_a], encodings_future_pred[ind_b])

                pred_matrix /= len(example_imgs) ** 2

                plt.figure(figsize=[10, 10])
                plt.imshow(pred_matrix, cmap='Oranges')
                plt.xticks(np.arange(10))
                plt.yticks(np.arange(10))
                plt.colorbar()
                plt.tight_layout()
                try:
                    plt.savefig(self.fig_dir / 'pred_matrix.png')
                except Exception as e:
                    print(e)
                plt.close()

            return True

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