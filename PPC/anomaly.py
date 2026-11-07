import numpy as np
import tensorflow as tf

from . import probability, wrappers

class Anomaly:

    def __init__(self, encoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]):
        """Init method for anomaly detection class.

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
