from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import tensorflow as tf
import numpy as np

if  __name__ == '__main__':

    np.random.seed(0)
    tf.keras.utils.set_random_seed(0)

    # Select GPU's
    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    # Select save directory
    save_dir = Path(__file__).parent.resolve() / 'models/model-001/'

    # If the model does not exist, create and train it
    if not save_dir.exists():

        # Settings
        settings = PPC.tools.Settings(
            batch_size           = 32,
            Ne                   = 16,
            Ng                   = 32,
            encoder_filters      = 8,
            forecast_dense_units = 64,
        )

        # Create dataset instances
        dataset = tools.SineFrequencyDataset(
            path       = Path(__file__).parent.resolve() / 'data.h5',
            dataset    = 'train',
            split      = np.array([0.8, 0.1, 0.1]),
            batch_size = settings.batch_size,
        )

        val_dataset = tools.SineFrequencyDataset(
            path       = Path(__file__).parent.resolve() / 'data.h5',
            dataset    = 'val',
            split      = np.array([0.8, 0.1, 0.1]),
            batch_size = settings.batch_size,
        )

        # Create models
        encoder_model = tools.models.encoder_model(
            Ne      = settings.Ne,
            filters = settings.encoder_filters,
            L       = dataset.L,
        )

        ar_model = tools.models.ar_model(
            Np = dataset.Np,
            Ne = settings.Ne,
            Ng = settings.Ng,
        )

        forecast_models = [tools.models.forecast_model(
            Ng          = settings.Ng,
            Ne          = settings.Ne,
            dense_units = settings.forecast_dense_units,
        ) for _ in range(dataset.Nf)]

        # Train models
        trainer = PPC.trainers.PCPC(
            encoder_model   = encoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.ContrastiveNLLRLoss(PPC.weights.Uniform(dataset.Nf), warmup=True),
            optimizer          = tf.keras.optimizers.Adam(learning_rate=1e-4),
            callbacks          = [PPC.callbacks.TerminateOnNaN()],
            iterations         = 5000,
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.ContrastiveNLLRLoss(PPC.weights.Uniform(dataset.Nf), warmup=False),
            optimizer          = tf.keras.optimizers.Adam(learning_rate=1e-4),
            callbacks          = [PPC.callbacks.TerminateOnNaN(), PPC.callbacks.EarlyStopping(monitor='val_forecast_loss', patience=10000)],
        )

        # Save models
        PPC.tools.save_pcpc_models(
            dir             = save_dir,
            encoder_model   = encoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        # Save settings
        settings.save(save_dir / 'settings.json')
