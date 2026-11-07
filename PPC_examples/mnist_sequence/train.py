from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import tensorflow as tf
import numpy as np

if  __name__ == '__main__':

    # Seed all random number generators (operations are not made deterministic: UpSampling2D has no deterministic GPU implementation)
    np.random.seed(0)
    tf.keras.utils.set_random_seed(0)

    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    batch_size           = 32
    Ne                   = 16
    Ng                   = Ne*2
    encoder_filters      = 32
    forecast_dense_units = Ng*2
    lambda_loss          = 1e4

    # Settings
    settings = PPC.tools.Settings(
        batch_size           = batch_size,
        Np                   = 1,
        Nf                   = 1,
        Ne                   = Ne,
        Ng                   = Ng,
        encoder_filters      = encoder_filters,
        forecast_dense_units = forecast_dense_units,
        lambda_loss          = lambda_loss,
        learning_rate        = 1e-4,
        warmup_iterations    = 1000,
        patience             = 10000,
    )

    print(settings)

    # Select save directory
    save_dir = Path(__file__).parent.resolve() / 'models/{}'.format(hash(settings))

    # Train the model if it has not been trained yet
    if not (save_dir / 'settings.json').exists():

        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset instances
        MNIST_inst     = tools.data.MNIST('train')
        val_MNIST_inst = tools.data.MNIST('val')

        dataset = tools.data.MNISTSequenceDataset(
            MNIST_inst   = MNIST_inst,
            batch_size   = settings.batch_size,
            Np           = settings.Np,
            Nf           = settings.Nf,
            augmentation = tools.augmentation.mnist_augmentation(),
            seed         = 0,
        )
        
        val_dataset = tools.data.MNISTSequenceDataset(
            MNIST_inst   = val_MNIST_inst,
            batch_size   = settings.batch_size,
            Np           = settings.Np,
            Nf           = settings.Nf,
            augmentation = None,
            seed         = 1,
        )

        # Create models
        encoder_model = tools.models.encoder_model(
            Ne          = settings.Ne,
            filters     = settings.encoder_filters,
            image_shape = dataset.image_shape,
        )

        decoder_model = tools.models.decoder_model(
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
        trainer = PPC.trainers.PPC(
            encoder_model   = encoder_model,
            decoder_model   = decoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        monitor_callback = tools.callbacks.MNISTMonitor(
            encoder_model           = encoder_model,
            decoder_model           = decoder_model,
            ar_model                = ar_model,
            forecast_models         = forecast_models,
            MNIST_inst              = val_MNIST_inst,
            eval_examples_per_digit = 3,
            eval_iterations         = 100,
            fig_dir                 = save_dir,
        )

        trainer.train(
            dataset                  = dataset,
            val_dataset              = val_dataset,
            forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.uniform(settings.Nf), warmup=True),
            reconstruction_loss_func = PPC.losses.MeanSquaredError(),
            lambda_loss              = settings.lambda_loss,
            optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            iterations               = settings.warmup_iterations,
            callbacks                = [
                PPC.callbacks.TerminateOnNaN(),
                monitor_callback,
            ]
        )
        
        trainer.train(
            dataset                  = dataset,
            val_dataset              = val_dataset,
            forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.uniform(settings.Nf), warmup=False),
            reconstruction_loss_func = PPC.losses.MeanSquaredError(),
            lambda_loss              = settings.lambda_loss,
            optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            callbacks                = [
                PPC.callbacks.TerminateOnNaN(), 
                PPC.callbacks.RestoreOnEarlyStopping(monitor='val_forecast_loss', patience=settings.patience, models=[encoder_model, decoder_model, ar_model] + forecast_models),
                monitor_callback,
            ],
        )

        # Save models
        PPC.tools.save_ppc_models(
            dir             = save_dir,
            encoder_model   = encoder_model,
            decoder_model   = decoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        # Save settings
        settings.save(save_dir / 'settings.json')