from typing import Optional, Callable, Tuple, List
import numpy as np
import tensorflow as tf

from . import wrappers

class PPC:
    """Trainer for Probabilistic Predictive Coding: encoder/decoder autoencoder combined with a probabilistic forecast of future encodings"""

    def __init__(self, encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]):
        """Init method for trainer class

        Args:
            encoder_model (tf.keras.Model): encoder model, [B, ...] -> [B, Ne]
            decoder_model (tf.keras.Model): decoder model, [B, Ne] -> [B, ...]
            ar_model (tf.keras.Model): autoregressive model, [B, Np, Ne] -> [B, Np, Ng]. GRU/LSTM option return_sequences must be set to True.
            forecast_models (list[tf.keras.Model]): list of Nf models, [B, Ng] -> [B, Ne, 2]
        """

        # Input arguments
        self.encoder_model   = encoder_model
        self.decoder_model   = decoder_model
        self.ar_model        = ar_model
        self.forecast_models = forecast_models

        # Create adapted models that can handle higher dimensionalities
        self.encoder_model_td1 = wrappers.TimeDistributed(self.encoder_model)        # [B, T1, ...] -> [B, T1, Ne]
        self.decoder_model_td1 = wrappers.TimeDistributed(self.decoder_model)        # [B, T1, Ne] -> [B, T1, ...]
        self.forecast_model_ts = wrappers.TimeStacked(self.forecast_models, [False]) # [B, Ng] -> [B, T3, Ne, 2]

        # Collect trainable weights variables for future use in gradient descent
        self.trainable_weights = self.encoder_model.trainable_weights + self.decoder_model.trainable_weights + self.ar_model.trainable_weights
        for forecast_model in self.forecast_models:
            self.trainable_weights += forecast_model.trainable_weights

    def train(self, dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, forecast_loss_func: Callable, reconstruction_loss_func: Callable, lambda_loss: float, optimizer: tf.keras.optimizers.Optimizer, callbacks: Optional[List[Callable]]=None, iterations: Optional[int]=None, loss_window: int=100):
        """Train the models with the provided parameters

        Args:
            dataset (tf.data.Dataset): train dataset
            val_dataset (tf.data.Dataset): validation dataset
            forecast_loss_func (Callable): forecast loss function
            reconstruction_loss_func (Callable): reconstruction loss function
            lambda_loss (float): loss weight for reconstruction loss
            optimizer (tf.keras.optimizers.Optimizer): optimizer
            callbacks (List[Callable]): callbacks
            iterations (Optional[int], optional): number of iterations to train. Defaults to None.
            loss_window (int, optional): number of values to consider in the moving average loss that is displayed. Defaults to 100.
        """

        # Prepare loss memory
        train_losses = {
            'forecast_loss': np.full(loss_window, np.nan, dtype=np.float32),
            'reconstruction_loss': np.full(loss_window, np.nan, dtype=np.float32),
            'total_loss': np.full(loss_window, np.nan, dtype=np.float32),
        }

        val_losses = {
            'forecast_loss': np.full(loss_window, np.nan, dtype=np.float32),
            'reconstruction_loss': np.full(loss_window, np.nan, dtype=np.float32),
            'total_loss': np.full(loss_window, np.nan, dtype=np.float32),
        }

        # Loop through train and validation datasets
        for i, ((samples_past, samples_future), (val_samples_past, val_samples_future)) in enumerate(zip(dataset, val_dataset)):

            # Apply a train step
            forecast_loss, reconstruction_loss, total_loss = self.train_step(samples_past, samples_future, forecast_loss_func, reconstruction_loss_func, lambda_loss, optimizer)

            # Add train loss to memory
            train_losses['forecast_loss']           = np.roll(train_losses['forecast_loss'], -1)
            train_losses['forecast_loss'][-1]       = forecast_loss
            train_losses['reconstruction_loss']     = np.roll(train_losses['reconstruction_loss'], -1)
            train_losses['reconstruction_loss'][-1] = reconstruction_loss
            train_losses['total_loss']              = np.roll(train_losses['total_loss'], -1)
            train_losses['total_loss'][-1]          = total_loss

            # Apply a validation step
            val_forecast_loss, val_reconstruction_loss, val_total_loss = self.test_step(val_samples_past, val_samples_future, forecast_loss_func, reconstruction_loss_func, lambda_loss)

            # Add validation loss to memory
            val_losses['forecast_loss']           = np.roll(val_losses['forecast_loss'], -1)
            val_losses['forecast_loss'][-1]       = val_forecast_loss
            val_losses['reconstruction_loss']     = np.roll(val_losses['reconstruction_loss'], -1)
            val_losses['reconstruction_loss'][-1] = val_reconstruction_loss
            val_losses['total_loss']              = np.roll(val_losses['total_loss'], -1)
            val_losses['total_loss'][-1]          = val_total_loss

            # Compile status string and print
            if iterations is None:
                status = 'iteration: {}'.format(i+1)
            else:
                status = 'iteration: {}/{}'.format(i+1, iterations)

            status += ' - forecast_loss: {:.2e}'.format(np.nanmean(train_losses['forecast_loss']))
            status += ' - rec_loss: {:.2e}'.format(np.nanmean(train_losses['reconstruction_loss']))
            status += ' - total_loss: {:.2e}'.format(np.nanmean(train_losses['total_loss']))

            status += ' - val_forecast_loss: {:.2e}'.format(np.nanmean(val_losses['forecast_loss']))
            status += ' - val_rec_loss: {:.2e}'.format(np.nanmean(val_losses['reconstruction_loss']))
            status += ' - val_total_loss: {:.2e}'.format(np.nanmean(val_losses['total_loss']))

            print('\r{}'.format(status), end='', flush=True)

            # Check callbacks
            if callbacks is not None:
                for callback in callbacks:
                    result = callback(
                        forecast_loss           = forecast_loss,
                        reconstruction_loss     = reconstruction_loss,
                        total_loss              = total_loss,
                        val_forecast_loss       = val_forecast_loss,
                        val_reconstruction_loss = val_reconstruction_loss,
                        val_total_loss          = val_total_loss,
                    )

                    if result is False:
                        print('Interrupted training by callback')
                        return

            # Check if training is completed
            if iterations is not None and i + 1 == iterations:
                print()
                print('Training completed')
                return

    def train_step(self, samples_past: tf.Tensor, samples_future: tf.Tensor, forecast_loss_func: Callable, reconstruction_loss_func: Callable, lambda_loss: float, optimizer: tf.keras.optimizers.Optimizer) -> Tuple[float, float, float]:
        """Apply a single train step

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            samples_future (tf.Tensor): future samples, shape [B, Nf, ...]
            forecast_loss_func (Callable): forecast loss function
            reconstruction_loss_func (Callable): reconstruction loss function
            optimizer (tf.keras.optimizers.Optimizer): optimizer

        Returns:
            Tuple[float, float, float]: forecast loss, reconstruction loss and total loss
        """

        # The models are called with training=True, so layers such as BatchNormalization use batch statistics and update their moving averages
        with tf.GradientTape() as tape:

            # Determine encodings for past and positive/negative future samples
            encodings_past        = self.encoder_model_td1(samples_past, training=True)   # [B, Np, Ne]
            encodings_future_true = self.encoder_model_td1(samples_future, training=True) # [B, Nf, Ne]

            # Predict future samples
            contexts_past         = self.ar_model(encodings_past, training=True)          # [B, Np, Ng]
            context_past          = contexts_past[:, -1, :]                               # [B, Ng]
            encodings_future_pred = self.forecast_model_ts(context_past, training=True)   # [B, Nf, Ne, 2]

            # Apply forecast loss function
            forecast_loss = forecast_loss_func(encodings_future_true, encodings_future_pred)

            # Decode encodings
            decodings_past   = self.decoder_model_td1(encodings_past, training=True)
            decodings_future = self.decoder_model_td1(encodings_future_true, training=True)

            # Apply reconstruction loss function
            reconstruction_loss = reconstruction_loss_func(samples_past, decodings_past, samples_future, decodings_future)

            # Determine total loss
            total_loss = forecast_loss + lambda_loss * reconstruction_loss

        # Fetch gradients and optimize all models
        grads = tape.gradient(total_loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return float(forecast_loss.numpy()), float(reconstruction_loss.numpy()), float(total_loss.numpy())

    def test_step(self, samples_past: tf.Tensor, samples_future: tf.Tensor, forecast_loss_func: Callable, reconstruction_loss_func: Callable, lambda_loss: float) -> Tuple[float, float, float]:
        """Apply a single test step

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            samples_future (tf.Tensor): future samples, shape [B, Nf, ...]
            forecast_loss_func (Callable): forecast loss function
            reconstruction_loss_func (Callable): reconstruction loss function

        Returns:
            Tuple[float, float, float]: forecast loss, reconstruction loss and total loss
        """

        # Determine encodings for past and positive/negative future samples
        encodings_past        = self.encoder_model_td1(samples_past, training=False)   # [B, Np, Ne]
        encodings_future_true = self.encoder_model_td1(samples_future, training=False) # [B, Nf, Ne]

        # Predict future samples
        contexts_past         = self.ar_model(encodings_past, training=False)          # [B, Np, Ng]
        context_past          = contexts_past[:, -1, :]                                # [B, Ng]
        encodings_future_pred = self.forecast_model_ts(context_past, training=False)   # [B, Nf, Ne, 2]

        # Apply forecast loss function
        forecast_loss = forecast_loss_func(encodings_future_true, encodings_future_pred)

        # Decode encodings
        decodings_past   = self.decoder_model_td1(encodings_past, training=False)
        decodings_future = self.decoder_model_td1(encodings_future_true, training=False)

        # Apply reconstruction loss function
        reconstruction_loss = reconstruction_loss_func(samples_past, decodings_past, samples_future, decodings_future)

        # Determine total loss
        total_loss = forecast_loss + lambda_loss * reconstruction_loss

        return float(forecast_loss.numpy()), float(reconstruction_loss.numpy()), float(total_loss.numpy())

class PCPC:
    """Trainer for Probabilistic Contrastive Predictive Coding: no decoder, the forecast distribution is trained contrastively against negative samples"""

    def __init__(self, encoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]):
        """Init method for trainer class

        Args:
            encoder_model (tf.keras.Model): encoder model, [B, ...] -> [B, Ne]
            ar_model (tf.keras.Model): autoregressive model, [B, Np, Ne] -> [B, Np, Ng]. GRU/LSTM option return_sequences must be set to True.
            forecast_models (list[tf.keras.Model]): list of Nf models, [B, Ng] -> [B, Ne, 2]
        """

        # Input arguments
        self.encoder_model   = encoder_model
        self.ar_model        = ar_model
        self.forecast_models = forecast_models

        # Create adapted models that can handle higher dimensionalities
        self.encoder_model_td1 = wrappers.TimeDistributed(self.encoder_model)        # [B, T1, ...] -> [B, T1, Ne]
        self.encoder_model_td2 = wrappers.TimeDistributed(self.encoder_model_td1)    # [B, T2, T1, ...] -> [B, T2, T1, Ne]
        self.forecast_model_ts = wrappers.TimeStacked(self.forecast_models, [False]) # [B, Ng] -> [B, T3, Ne, 2]

        # Collect trainable weights variables for future use in gradient descent
        self.trainable_weights = self.encoder_model.trainable_weights + self.ar_model.trainable_weights
        for forecast_model in self.forecast_models:
            self.trainable_weights += forecast_model.trainable_weights

    def train(self, dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, forecast_loss_func: Callable, optimizer: tf.keras.optimizers.Optimizer, callbacks: Optional[List[Callable]]=None, iterations: Optional[int]=None, loss_window: int=100):
        """Train the models with the provided parameters

        Args:
            dataset (tf.data.Dataset): train dataset
            val_dataset (tf.data.Dataset): validation dataset
            forecast_loss_func (Callable): contrastive forecast loss function, e.g. losses.ContrastiveNLLRLoss
            optimizer (tf.keras.optimizers.Optimizer): optimizer
            callbacks (List[Callable]): callbacks
            iterations (Optional[int], optional): number of iterations to train. Defaults to None.
            loss_window (int, optional): number of values to consider in the moving average loss that is displayed. Defaults to 100.
        """

        # Prepare loss memory
        train_losses = {
            'forecast_loss': np.full(loss_window, np.nan, dtype=np.float32),
        }

        val_losses = {
            'forecast_loss': np.full(loss_window, np.nan, dtype=np.float32),
        }

        # Loop through train and validation datasets
        for i, ((samples_past, samples_future_pos, samples_future_neg), (val_samples_past, val_samples_future_pos, val_samples_future_neg)) in enumerate(zip(dataset, val_dataset)):

            # Apply a train step
            forecast_loss = self.train_step(samples_past, samples_future_pos, samples_future_neg, forecast_loss_func, optimizer)

            # Add train loss to memory
            train_losses['forecast_loss']     = np.roll(train_losses['forecast_loss'], -1)
            train_losses['forecast_loss'][-1] = forecast_loss

            # Apply a validation step
            val_forecast_loss = self.test_step(val_samples_past, val_samples_future_pos, val_samples_future_neg, forecast_loss_func)

            # Add validation loss to memory
            val_losses['forecast_loss']     = np.roll(val_losses['forecast_loss'], -1)
            val_losses['forecast_loss'][-1] = val_forecast_loss

            # Compile status string and print
            if iterations is None:
                status = 'iteration: {}'.format(i+1)
            else:
                status = 'iteration: {}/{}'.format(i+1, iterations)

            status += ' - forecast_loss: {:.2e}'.format(np.nanmean(train_losses['forecast_loss']))
            status += ' - val_forecast_loss: {:.2e}'.format(np.nanmean(val_losses['forecast_loss']))

            print('\r{}'.format(status), end='', flush=True)

            # Check callbacks
            if callbacks is not None:
                for callback in callbacks:
                    result = callback(
                        forecast_loss     = forecast_loss,
                        val_forecast_loss = val_forecast_loss,
                    )

                    if result is False:
                        print('Interrupted training by callback')
                        return

            # Check if training is completed
            if iterations is not None and i + 1 == iterations:
                print()
                print('Training completed')
                return

    def train_step(self, samples_past: tf.Tensor, samples_future_pos: tf.Tensor, samples_future_neg: tf.Tensor, forecast_loss_func: Callable, optimizer: tf.keras.optimizers.Optimizer) -> float:
        """Apply a single train step

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            samples_future_pos (tf.Tensor): positive future samples, shape [B, Nf, ...]
            samples_future_neg (tf.Tensor): negative future samples, shape [B, Nchi, Nf, ...]
            forecast_loss_func (Callable): forecast loss function
            optimizer (tf.keras.optimizers.Optimizer): optimizer

        Returns:
            float: forecast loss
        """

        # The models are called with training=True, so layers such as BatchNormalization use batch statistics and update their moving averages
        with tf.GradientTape() as tape:

            # Determine encodings for past and positive/negative future samples
            encodings_past        = self.encoder_model_td1(samples_past, training=True)       # [B, Np, Ne]
            encodings_future_pos  = self.encoder_model_td1(samples_future_pos, training=True) # [B, Nf, Ne]
            encodings_future_neg  = self.encoder_model_td2(samples_future_neg, training=True) # [B, Nchi, Nf, Ne]

            # Predict future samples
            contexts_past         = self.ar_model(encodings_past, training=True)              # [B, Np, Ng]
            context_past          = contexts_past[:, -1, :]                                   # [B, Ng]
            encodings_future_pred = self.forecast_model_ts(context_past, training=True)       # [B, Nf, Ne, 2]

            # Apply loss function
            forecast_loss = forecast_loss_func(encodings_future_pos, encodings_future_neg, encodings_future_pred)

        # Fetch gradients and optimize all models
        grads = tape.gradient(forecast_loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return float(forecast_loss.numpy())

    def test_step(self, samples_past: tf.Tensor, samples_future_pos: tf.Tensor, samples_future_neg: tf.Tensor, forecast_loss_func: Callable) -> float:
        """Apply a single test step

        Args:
            samples_past (tf.Tensor): past samples, shape [B, Np, ...]
            samples_future_pos (tf.Tensor): positive future samples, shape [B, Nf, ...]
            samples_future_neg (tf.Tensor): negative future samples, shape [B, Nchi, Nf, ...]
            forecast_loss_func (Callable): forecast loss function

        Returns:
            float: forecast loss
        """

        # Determine encodings for past and positive/negative future samples
        encodings_past        = self.encoder_model_td1(samples_past, training=False)       # [B, Np, Ne]
        encodings_future_pos  = self.encoder_model_td1(samples_future_pos, training=False) # [B, Nf, Ne]
        encodings_future_neg  = self.encoder_model_td2(samples_future_neg, training=False) # [B, Nchi, Nf, Ne]

        # Predict future samples
        contexts_past         = self.ar_model(encodings_past, training=False)              # [B, Np, Ng]
        context_past          = contexts_past[:, -1, :]                                    # [B, Ng]
        encodings_future_pred = self.forecast_model_ts(context_past, training=False)       # [B, Nf, Ne, 2]

        # Apply loss function
        forecast_loss = forecast_loss_func(encodings_future_pos, encodings_future_neg, encodings_future_pred)

        return float(forecast_loss.numpy())
