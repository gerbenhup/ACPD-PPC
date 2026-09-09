#########
## Anomaly Detection Using Probabilistic Predictive Coding
## Roelof G. Hup, 10 November 2023
## Eindhoven University of Technology
#########

from typing import Optional, Callable, Tuple, List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
import sklearn.decomposition
import json
import sys
import subprocess
import hashlib

class trainers:

    class PPC:

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

            with tf.GradientTape() as tape:
                
                # Determine encodings for past and positive/negative future samples
                encodings_past        = self.encoder_model_td1(samples_past)   # [B, Np, Ne]
                encodings_future_true = self.encoder_model_td1(samples_future) # [B, Nf, Ne]

                # Predict future samples
                contexts_past         = self.ar_model(encodings_past)          # [B, Np, Ng]
                context_past          = contexts_past[:, -1, :]                # [B, Ng]
                encodings_future_pred = self.forecast_model_ts(context_past)   # [B, Nf, Ne, 2]
                
                # Apply forecast loss function
                forecast_loss = forecast_loss_func(encodings_future_true, encodings_future_pred)

                # Decode encodings
                decodings_past   = self.decoder_model_td1(encodings_past)
                decodings_future = self.decoder_model_td1(encodings_future_true)

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
            encodings_past        = self.encoder_model_td1(samples_past)   # [B, Np, Ne]
            encodings_future_true = self.encoder_model_td1(samples_future) # [B, Nf, Ne]

            # Predict future samples
            contexts_past         = self.ar_model(encodings_past)          # [B, Np, Ng]
            context_past          = contexts_past[:, -1, :]                # [B, Ng]
            encodings_future_pred = self.forecast_model_ts(context_past)   # [B, Nf, Ne, 2]
            
            # Apply forecast loss function
            forecast_loss = forecast_loss_func(encodings_future_true, encodings_future_pred)

            # Decode encodings
            decodings_past   = self.decoder_model_td1(encodings_past)
            decodings_future = self.decoder_model_td1(encodings_future_true)

            # Apply reconstruction loss function
            reconstruction_loss = reconstruction_loss_func(samples_past, decodings_past, samples_future, decodings_future)

            # Determine total loss
            total_loss = forecast_loss + lambda_loss * reconstruction_loss

            return float(forecast_loss.numpy()), float(reconstruction_loss.numpy()), float(total_loss.numpy())

class callbacks:

    class RestoreOnEarlyStopping:
        """Training callback that restores model weights to their best point and stops training if stagnant"""

        def __init__(self, monitor: str, patience: int, models: List[tf.keras.Model]):
            """
            Args:
                monitor (str): name of the loss to monitor, as passed to `__call__`
                patience (int): number of iterations without improvement before stopping training
                models (List[tf.keras.Model]): models whose weights are tracked and restored on stopping
            """

            self.monitor           = monitor
            self.patience          = patience
            self.models            = models
            self.best_weights      = [model.get_weights() for model in self.models]
            self.best_loss         = np.inf
            self.best_iteration    = 0
            self.current_iteration = 0

        def __call__(self, **kwargs) -> bool:
            """Check the monitored loss for the current iteration, tracking or restoring the best weights

            Args:
                **kwargs: named loss values for the current iteration, must include the monitored loss name

            Returns:
                bool: False once patience is exceeded (interrupts training), True otherwise
            """

            if self.monitor not in kwargs:
                raise ValueError('Monitor {} not provided'.format(self.monitor))

            self.current_iteration += 1

            if kwargs[self.monitor] < self.best_loss:
                self.best_loss      = kwargs[self.monitor]
                self.best_iteration = self.current_iteration
                self.best_weights   = [model.get_weights() for model in self.models]
                return True

            elif self.current_iteration - self.best_iteration > self.patience:
                for model, weights in zip(self.models, self.best_weights):
                    model.set_weights(weights)

                print()
                print('Early Stopping triggered')
                return False

    class TerminateOnNaN:
        """Training callback that stops training as soon as any reported loss becomes NaN"""

        def __init__(self):
            pass

        def __call__(self, **kwargs) -> bool:
            """Check the losses reported for the current iteration for NaN values

            Args:
                **kwargs: named loss values for the current iteration

            Returns:
                bool: False if any loss is NaN (interrupts training), True otherwise
            """

            for name, value in kwargs.items():
                if np.any(np.isnan(value)):
                    print()
                    print('Terminate on NaN triggered by {}'.format(name))
                    return False
                
            return True

class Anomaly:

    def __init__(self, encoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]):
        """Init method for anomaly detection class.

        Args:
            encoder_model (tf.keras.Model): encoder model, [B, ...] -> [B, Ne]
            ar_model (tf.keras.Model): autoregressive model, [B, Np, Ne] -> [B, Np, Ng]. GRU/LSTM option return_sequences must be set to True.
            forecast_models (list[tf.keras.Model]): list of Nf models, ([B, Ng], [B, Na]) -> [B, Ne, 2]
        """
        
        # Input arguments
        self.encoder_model   = encoder_model
        self.ar_model        = ar_model
        self.forecast_models = forecast_models

        # Create adapted models that can handle higher dimensionalities
        self.encoder_model_td1 = wrappers.TimeDistributed(self.encoder_model)

    def __call__(self, samples: tf.Tensor) -> tf.Tensor:
        """Call function for anomaly detection class.

        Args:
            samples (tf.Tensor): samples, shape [B, Ns, ...]

        Returns:
            tf.Tensor: Tail probabilities for every time step, forecast step and anomaly class, shape [B, Ns, Nf]
        """

        # Infer encodings
        encodings_true = self.encoder_model_td1(samples) # [B, Ns, Ne]

        # Infer context vectors, based on AR model parameter
        Np       = self.ar_model.input.shape[1]
        contexts = tf.map_fn(
            fn                  = lambda ti: self.ar_model(encodings_true[:, ti:ti+Np, :])[:, -1, :],
            elems               = tf.range(encodings_true.shape[1]-Np, dtype=tf.int32),
            fn_output_signature = tf.TensorSpec((encodings_true.shape[0], self.ar_model.output.shape[-1]), dtype=tf.float32)
        ) # [Ns-Np, B, Ng]
        contexts = tf.transpose(contexts, [1, 0, 2]) # [B, Ns-Np, Ng]

        # Infer forecasts
        encodings_pred = []
        for fmi in range(len(self.forecast_models)):
            encodings_pred.append([])
            for ti in range(contexts.shape[1]):
                # time step of predicted encoding: ti + Np + fmi
                encodings_pred[-1].append(
                    self.forecast_models[fmi](contexts[:, ti, :]) # [B, Ne, 2]
                )
            encodings_pred[-1] = tf.stack(encodings_pred[-1], axis=0) # [Ns-Np, B, Ne, 2]
        encodings_pred = tf.stack(encodings_pred, axis=0) # [Nf, Ns-Np, B, Ne, 2]
        encodings_pred = tf.transpose(encodings_pred, (0, 2, 1, 3, 4)) # [Nf, B, Ns-Np, Ne, 2]

        # Window function responsible for rolling over the forecast model axis
        def evaluate(fmi):

            # Align encodings and forecasts
            encodings_true_local = encodings_true[:, (fmi+Np):, :] # [B, Ns-fmi-Np, Ne]

            if fmi == 0:
                encodings_pred_local = encodings_pred[fmi, :, :, :, :] # [B, Ns-fmi-Np, Ne, 2]
            else:
                encodings_pred_local = encodings_pred[fmi, :, :-fmi, :, :] # [B, Ns-fmi-Np, Ne, 2]

            # Determine tail probabilities
            probs = probability.MultiVariateGaussianTailProbability(encodings_true_local, encodings_pred_local) # [B, Ns-fmi-Np]

            # Pad steps that are not forecasted with NaNs
            probs = tf.pad(probs, [[0, 0], [fmi+Np, 0]], constant_values=np.nan) # [B, Ns]

            return probs

        # Apply window function
        probs = tf.map_fn(
            fn                  = evaluate,
            elems               = tf.range(len(self.forecast_models), dtype=tf.int32),
            fn_output_signature = tf.TensorSpec((encodings_true.shape[0], encodings_true.shape[1]), dtype=tf.float32)
        ) # [Nf, B, Ns]
        
        # Transpose to correct shape
        probs = tf.transpose(probs, [1, 2, 0]) # [B, Ns, Nf]

        return probs

class losses:
    
    class MeanSquaredError:
        """Mean squared error reconstruction loss"""

        def __init__(self):
            pass

        def __call__(self, samples_past: tf.Tensor, decodings_past: tf.Tensor, samples_future: tf.Tensor, decodings_future: tf.Tensor) -> tf.Tensor:
            """Call function

            Args:
                samples_past (tf.Tensor): Samples in past, [B, Np, ...]
                decodings_past (tf.Tensor): Decoded encodings in past, [B, Np, ...]
                samples_future (tf.Tensor): Samples in future, [B, Nf, ...]
                decodings_future (tf.Tensor): Decoded encodings in future, [B, Nf, ...]

            Returns:
                tf.Tensor: loss values
            """

            samples   = tf.concat([samples_past, samples_future], axis=1)
            decodings = tf.concat([decodings_past, decodings_future], axis=1)

            return tf.math.reduce_mean(tf.math.squared_difference(samples, decodings))

    class MaximumLogLikelihoodLoss:

        def __init__(self, omega: tf.Tensor, warmup: bool):
            """Init function

            Args:
                omega (tf.Tensor): weights of losses for each time step
                warmup (bool): whether to assume unit standard deviation (True) or not (False)
            """

            self.omega = omega

            if warmup:
                self.loss_func = probability.GaussianLogLikelihoodUnitVariance
            else:
                self.loss_func = probability.GaussianLogLikelihood

        def __call__(self, encodings_future_true: tf.Tensor, encodings_future_pred: tf.Tensor) -> tf.Tensor:
            """Call function

            Args:
                encodings_future_true (tf.Tensor): encodings of future true samples (shape [B, Nf, Ne])
                encodings_future_pred (tf.Tensor): encodings of future predicted samples (shape [B, Nf, Ne, 2])

            Returns:
                tf.Tensor: loss values
            """

            losses        = -self.loss_func(encodings_future_true, encodings_future_pred) # [B, Nf]
            batch_loss    = tf.math.reduce_mean(losses, axis=0) # [Nf,]
            weighted_loss = tf.tensordot(batch_loss, self.omega, 1) # []

            return weighted_loss

class dimension:

    class PCA:
        """Thin wrapper around scikit-learn's PCA, extended to also project Gaussian forecast distributions"""

        def __init__(self, n_components: int):
            """
            Args:
                n_components (int): dimensionality of the embedded space
            """

            self.n_components = n_components
            self.pca          = sklearn.decomposition.PCA(self.n_components)

        def fit(self, encodings: np.ndarray) -> None:
            """Fit PCA on the given encodings

            Args:
                encodings (np.ndarray): encodings to fit on, shape [N, Ne]
            """

            self.pca.fit(encodings)

        def fit_transform(self, encodings: np.ndarray) -> np.ndarray:
            """Fit PCA on the given encodings and return their low-dimensional projection

            Args:
                encodings (np.ndarray): encodings to fit on and project, shape [N, Ne]

            Returns:
                np.ndarray: projected encodings, shape [N, n_components]
            """

            return self.pca.fit_transform(encodings)

        def transform(self, encodings: np.ndarray) -> np.ndarray:
            """Project encodings onto the previously fitted principal components

            Args:
                encodings (np.ndarray): encodings to project, shape [N, Ne]

            Returns:
                np.ndarray: projected encodings, shape [N, n_components]
            """

            return self.pca.transform(encodings)

        def transform_gaussians(self, encodings_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Project Gaussian forecast distributions onto the previously fitted principal components

            Args:
                encodings_pred (np.ndarray): predicted encoding distributions, shape [N, Ne, 2] (means and standard deviations)

            Returns:
                Tuple[np.ndarray, np.ndarray]: projected means, shape [N, n_components], and projected covariance matrices, shape [N, n_components, n_components]
            """

            reduced_encodings_pred_means = self.pca.transform(encodings_pred[..., 0])
            reduced_encodings_pred_covs  = np.empty((len(encodings_pred), self.n_components, self.n_components), dtype=encodings_pred.dtype)

            for i in range(len(encodings_pred)):
                reduced_encodings_pred_covs[i] = np.matmul(np.matmul(self.pca.components_, np.diagflat(np.square(encodings_pred[i, :, 1]))), self.pca.components_.T)

            return reduced_encodings_pred_means, reduced_encodings_pred_covs

class probability:

    def GaussianLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
        """Gaussian log likelihood function

        Args:
            encodings_true (tf.Tensor): true encodings
            encodings_pred (tf.Tensor): predicted encoding distributions

        Returns:
            tf.Tensor: Gaussian log likelihood
        """        

        return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - tf.math.reduce_sum(math.log(encodings_pred[..., 1]), axis=-1) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)    

    def GaussianLogLikelihoodUnitVariance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
        """Gaussian log likelihood function which assumes unit variance

        Args:
            encodings_true (tf.Tensor): true encodings
            encodings_pred (tf.Tensor): predicted encoding distributions

        Returns:
            tf.Tensor: Gaussian log likelihood
        """

        return -encodings_pred.shape[-2]*math.log(tf.math.sqrt(2*np.pi)) - 1/2 * tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]), axis=-1)

    def GaussianExpNormLogLikelihood(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
        """Gaussian likelihood function with exp-log trick for numerical stability

        Args:
            encodings_true (tf.Tensor): true encodings
            encodings_pred (tf.Tensor): predicted encoding distributions

        Returns:
            tf.Tensor: Gaussian likelihood
        """

        loglikelihood  = probability.GaussianLogLikelihood(encodings_true, encodings_pred)
        loglikelihood -= tf.math.reduce_max(loglikelihood)
        likelihood     = math.exp(loglikelihood)
        
        return likelihood

    def SquaredMahalanobisDistance(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
        """Mahalanobis distance

        Args:
            z_hat (tf.Tensor): multivariate gaussian distribution parameters, shape [..., Ne, 2]. The last channel contains means and standard deviations.
            z (tf.Tensor): sample from multivariate gaussian distribution

        Returns:
            tf.Tensor: squared Mahalanobis distance
        """

        return tf.math.reduce_sum(math.square(encodings_true - encodings_pred[..., 0]) / math.square(encodings_pred[..., 1]), axis=-1)
    
    def ChiSquaredCDF(x: tf.Tensor, df: int) -> tf.Tensor:
        """Cumulative distribution function for the Chi-squared distribution.

        Args:
            x (tf.Tensor): input values
            df (int): number of degrees of freedom (in the context of CPCP: the encoding vector size)

        Returns:
            tf.Tensor: CDF values
        """        

        return tfp.distributions.Chi2(df).cdf(x)
    
    def MultiVariateGaussianTailProbability(encodings_true: tf.Tensor, encodings_pred: tf.Tensor) -> tf.Tensor:
        """Calculate the tail probability given a multivariate gaussian distribution and a sample.

        Args:
            z_hat (tf.Tensor): multivariate gaussian distribution parameters, shape [..., Ne, 2]. The last channel contains means and standard deviations.
            z (tf.Tensor): sample from multivariate gaussian distribution

        Returns:
            tf.Tensor: tail probabilities
        """        

        # Determine squared Mahalanobis distances
        dist = probability.SquaredMahalanobisDistance(encodings_true, encodings_pred)

        # Determine associated CDF values for the Chi-squared distribution
        cdf = probability.ChiSquaredCDF(dist, encodings_pred.shape[-2])

        # Convert to tail probability and return
        return 1. - cdf
    
class wrappers:

    def TimeDistributed(module: tf.keras.layers.Layer) -> tf.keras.Model:
        """Wrap module in TimeDistributed wrapper.

        Args:
            module (tf.keras.layers.Layer): module to wrap

        Returns:
            tf.keras.Model: wrapped module
        """

        # Get old and new input shapes
        input_shape     = list(module.input.shape[1:])
        new_input_shape = [None,] + input_shape

        # Create new wrapped module
        input      = tf.keras.Input(new_input_shape)
        output     = tf.keras.layers.TimeDistributed(module)(input)
        new_module = tf.keras.Model(input, output)

        return new_module
    
    def TimeStacked(modules: list[tf.keras.layers.Layer], inputs_concat: list[bool]) -> tf.keras.Model:
        """Wrap multiple modules with the same input sizes into one module with stacked outputs

        Args:
            modules (list[tf.keras.layers.Layer]): list of TensorFlow modules, e.g., models or layers
            inputs_concat (list[bool]): whether to stack the input elements of each seperate module or to use a single input

        Returns:
            tf.keras.Model: stacked modules in a single model
        """        

        new_inputs        = [None for _ in range(len(inputs_concat))]
        inputs_per_module = [[] for _ in range(len(modules))]

        # Create new inputs
        for i, (input_, concat) in enumerate(zip(modules[0].inputs, inputs_concat)):
            if concat is True:
                new_inputs[i] = tf.keras.Input((len(modules),) + input_.shape[1:])
            else:
                new_inputs[i] = tf.keras.Input(input_.shape[1:])

        # Determine inputs for every module
        for j in range(len(modules)):
            for i, (input_, concat) in enumerate(zip(new_inputs, inputs_concat)):
                if concat is True:
                    inputs_per_module[j].append(tf.keras.layers.Lambda(lambda x: x[:, j])(input_))
                else:
                    inputs_per_module[j].append(input_)
        
        # Fetch module outputs and stack them in time
        iresults = [module(inputs) for module, inputs in zip(modules, inputs_per_module)]
        sresults = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(iresults)

        # Create new module
        new_module = tf.keras.Model(new_inputs, sresults)
        
        return new_module

class weights:

    def Uniform(Nf: int) -> tf.Tensor:
        """Generate uniform time weights for MaximumLogLikelihoodLoss

        Args:
            Nf (int): number of future predictions

        Returns:
            tf.Tensor: uniform weights, each equal to 1/Nf
        """

        omega = np.ones(Nf, dtype=np.float32) / Nf
        omega = tf.constant(omega)

        return omega

class math:

    def log(x: tf.Tensor) -> tf.Tensor:
        """Safe logarithm function. Inputs are clipped to avoid non-allowed input values.
        Clipped at (almost) zero. Gradients not passing whenever input is not safe.

        Args:
            x (tf.Tensor): input value

        Returns:
            tf.Tensor: safe logarithm of input value
        """

        x_clipped = tf.math.maximum(x, 1.1754943e-38)
        x_log     = tf.math.log(x_clipped)

        return x_log

    def exp(x: tf.Tensor) -> tf.Tensor:
        """Safe exponent function. Inputs are clipped to avoid non-allowed input values.
        Clipped at log(tf.float32.max). Gradients not passing whenever input is not safe.

        Args:
            x (tf.Tensor): input value

        Returns:
            tf.Tensor: safe exponent of input value
        """        

        x_clipped = tf.math.minimum(x, 88.7228)
        x_exp     = tf.math.exp(x_clipped)

        return x_exp
    
    def square(x: tf.Tensor) -> tf.Tensor:
        """Safe square function. Inputs are clipped to avoid non-allowed input values.
        Clipped at sqrt(tf.float32.max). Gradients not passing whenever input is not safe.

        Args:
            x (tf.Tensor): input value

        Returns:
            tf.Tensor: safe square of input value
        """

        x_abs     = tf.math.abs(x)
        x_clipped = tf.math.minimum(x_abs, 1.8446743e+19)
        x_square  = tf.math.square(x_clipped)

        return x_square

class tools:

    class Settings:
        """Bag of experiment hyperparameters, savable/loadable as JSON and stably hashable for directory naming"""

        def __init__(self, **kwargs):
            """
            Args:
                **kwargs: hyperparameter names and values, stored as attributes
            """

            for name, value in kwargs.items():
                setattr(self, name, value)

        def save(self, file: Path):
            """Save the settings to a JSON file

            Args:
                file (Path): file to save the settings to
            """

            with open(file, 'w') as f:
                json.dump(vars(self), f)

        def load(file: Path):
            """Load settings from a JSON file

            Args:
                file (Path): file to load the settings from

            Returns:
                Settings: loaded settings
            """

            with open(file, 'r') as f:
                kwargs = json.load(f)

            return tools.Settings(**kwargs)

        def __str__(self):

            return str(vars(self))

        def __hash__(self):
            """Compute a stable hash of the settings, suitable for use as a model directory name

            Returns:
                int: stable hash, deterministic across interpreter runs (unlike Python's built-in string hash())
            """

            digest = hashlib.sha256(json.dumps(vars(self), sort_keys=True).encode()).hexdigest()

            return int(digest, 16) % (2**63)

    def save_model(model: tf.keras.Model, dir: Path, name: str):
        """Save a single model's architecture (JSON) and weights (H5) to a directory

        Args:
            model (tf.keras.Model): model to save
            dir (Path): directory to save the model in
            name (str): base filename to save the model as
        """

        with open(dir / '{}.json'.format(name), 'w') as f:
            f.write(model.to_json())

        model.save_weights(dir / '{}.h5'.format(name))

    def load_model(dir: Path, name: str) -> tf.keras.Model:
        """Load a single model's architecture (JSON) and weights (H5) from a directory

        Args:
            dir (Path): directory the model was saved in
            name (str): base filename the model was saved as

        Returns:
            tf.keras.Model: loaded model
        """

        with open(dir / '{}.json'.format(name), 'r') as f:
            model = tf.keras.models.model_from_json(f.read(), {'PPC': sys.modules[__name__]})
        
        model.load_weights(dir / '{}.h5'.format(name))

        return model
    
    def save_ppc_models(dir: Path, encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]) -> None:
        """Save all models of a PPC pipeline (encoder, decoder, autoregressive and forecast models) to a directory

        Args:
            dir (Path): directory to save the models in
            encoder_model (tf.keras.Model): encoder model
            decoder_model (tf.keras.Model): decoder model
            ar_model (tf.keras.Model): autoregressive model
            forecast_models (list[tf.keras.Model]): forecast models
        """

        dir.mkdir(parents=True, exist_ok=True)

        with open(dir / 'pcpc_models.txt', 'w') as f:

            tools.save_model(encoder_model, dir, 'encoder_model')
            f.write('encoder_model\n')
            tools.save_model(decoder_model, dir, 'decoder_model')
            f.write('decoder_model\n')
            tools.save_model(ar_model, dir, 'ar_model')
            f.write('ar_model\n')

            for i, forecast_model in enumerate(forecast_models):
                tools.save_model(forecast_model, dir, 'forecast_model_{}'.format(i))
                f.write('forecast_model_{}\n'.format(i))

    def load_ppc_models(dir: Path) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, list[tf.keras.Model]]:
        """Load all models of a PPC pipeline (encoder, decoder, autoregressive and forecast models) from a directory

        Args:
            dir (Path): directory the models were saved in

        Returns:
            Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, list[tf.keras.Model]]: encoder, decoder, autoregressive and forecast models
        """

        with open(dir / 'pcpc_models.txt', 'r') as f:
            models = f.read().strip().split('\n')

        forecast_models = []

        for line in models:
            if line == 'encoder_model':
                encoder_model = tools.load_model(dir, 'encoder_model')
            elif line == 'decoder_model':
                decoder_model = tools.load_model(dir, 'decoder_model')
            elif line == 'ar_model':
                ar_model = tools.load_model(dir, 'ar_model')
            elif 'forecast' in line:
                forecast_models.append(tools.load_model(dir, line))

        return encoder_model, decoder_model, ar_model, forecast_models

class gpu:

    def find_available(num_gpus: int, max_utilization: int, min_free_memory: int) -> list[int]:
        """Find GPU indices with sufficiently low utilization and sufficient free memory, via `nvidia-smi`

        Args:
            num_gpus (int): number of available GPU's to find
            max_utilization (int): maximum utilization (%) for a GPU to be considered available
            min_free_memory (int): minimum free memory (MiB) for a GPU to be considered available

        Returns:
            list[int]: indices of the first `num_gpus` GPU's that satisfy the given constraints
        """

        stats = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv')
        stats = stats.decode().splitlines()[1:]
        stats = [x.split(', ') for x in stats]
        stats = [[int(y[0][:-2]), int(y[1][:-4])] for y in stats]

        selected_gpus = []

        for gpu_id, stat in enumerate(stats):
            if stat[0] <= max_utilization and stat[1] >= min_free_memory:
                selected_gpus.append(gpu_id)
                if len(selected_gpus) == num_gpus:
                    break
        else:
            raise RuntimeError('Could not find available GPUs')

        return selected_gpus

    def select_gpus(selected_gpus: list[int], memory_growth: bool=False):
        """Restrict TensorFlow to the given GPU's for this session

        Args:
            selected_gpus (list[int]): indices of GPU's to make visible to TensorFlow
            memory_growth (bool, optional): whether to allow GPU memory growth instead of allocating all memory upfront. Defaults to False.
        """

        try:
            # Fetch available physical GPU's
            physical_gpus = tf.config.list_physical_devices('GPU')

            # Remove the GPU's not provided as input
            for i in reversed(range(len(physical_gpus))):
                if i not in selected_gpus:
                    physical_gpus.pop(i)
            
            # Set the selected GPU's as the visible devices for this session
            tf.config.set_visible_devices(physical_gpus, 'GPU')

            # Set memory growth
            if memory_growth is True:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        except Exception as e:
            print(e)
