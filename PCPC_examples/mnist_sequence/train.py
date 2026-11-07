from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import tensorflow as tf
import numpy as np

if  __name__ == '__main__':

    # Seed all random number generators and make TensorFlow operations deterministic
    np.random.seed(0)
    tf.keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()

    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    # Settings
    settings = PPC.tools.Settings(
        batch_size           = 32,
        Np                   = 1,
        Nf                   = 1,
        Nchi                 = 9,
        Ne                   = 16,
        Ng                   = 16,
        encoder_filters      = 64,
        forecast_dense_units = 128,
    )

    print(settings)

    # Select save directory
    save_dir = Path(__file__).parent.resolve() / 'models/model-001/'

    # If the model does not exist, create and train it
    if not save_dir.exists():

        # Create dataset instances
        MNIST_inst     = tools.data.MNIST('train')
        val_MNIST_inst = tools.data.MNIST('val')

        dataset = tools.data.MNISTSequenceDataset(
            MNIST_inst   = MNIST_inst,
            batch_size   = settings.batch_size,
            Np           = settings.Np,
            Nf           = settings.Nf,
            Nchi         = settings.Nchi,
            augmentation = tools.augmentation.MNISTaugmentation(),
            seed         = 0,
        )

        val_dataset = tools.data.MNISTSequenceDataset(
            MNIST_inst   = val_MNIST_inst,
            batch_size   = settings.batch_size,
            Np           = settings.Np,
            Nf           = settings.Nf,
            Nchi         = settings.Nchi,
            augmentation = None,
            seed         = 1,
        )

        # Create models
        encoder_model = tools.models.encoder_model(
            Ne          = settings.Ne,
            filters     = settings.encoder_filters,
            image_shape = dataset.image_shape,
        )

        ar_model = tools.models.ar_model(
            Np = settings.Np,
            Ne = settings.Ne,
            Ng = settings.Ng,
        )

        forecast_models = [tools.models.forecast_model(
            Ng          = settings.Ng,
            Ne          = settings.Ne,
            dense_units = settings.forecast_dense_units,
        ) for _ in range(settings.Nf)]

        # Train models
        trainer = PPC.trainers.PCPC(
            encoder_model   = encoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.ContrastiveNLLRLoss(PPC.weights.Uniform(settings.Nf), warmup=True),
            optimizer          = tf.keras.optimizers.Adam(learning_rate=1e-4),
            iterations         = 5000,
            callbacks          = [PPC.callbacks.TerminateOnNaN()],
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.ContrastiveNLLRLoss(PPC.weights.Uniform(settings.Nf), warmup=False),
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
