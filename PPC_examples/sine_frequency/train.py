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

    settings = PPC.tools.Settings(
        batch_size           = 32,
        Ne                   = 16,
        Ng                   = 32,
        encoder_filters      = 32,
        forecast_dense_units = 64,
        lambda_loss          = 1000.0,
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

        PositiveSample_inst     = tools.data.PositiveSamples(Path(__file__).parent / 'train_dataset.h5')
        val_PositiveSample_inst = tools.data.PositiveSamples(Path(__file__).parent / 'val_dataset.h5')

        # Create dataset instances
        dataset = tools.data.PositiveSamplesDataset(
            PositiveSample_inst = PositiveSample_inst,
            batch_size          = settings.batch_size,
            seed                = 0,
        )

        val_dataset = tools.data.PositiveSamplesDataset(
            PositiveSample_inst = val_PositiveSample_inst,
            batch_size          = settings.batch_size,
            seed                = 1,
        )

        # Create models
        encoder_model = tools.models.encoder_model(
            Ne      = settings.Ne,
            filters = settings.encoder_filters,
            L       = PositiveSample_inst.L,
        )

        decoder_model = tools.models.decoder_model(
            Ne      = settings.Ne,
            filters = settings.encoder_filters,
            L       = PositiveSample_inst.L,
        )

        ar_model = tools.models.ar_model(
            Np = PositiveSample_inst.Np,
            Ne = settings.Ne,
            Ng = settings.Ng,
        )

        forecast_models = [tools.models.forecast_model(
            Ng          = settings.Ng,
            Ne          = settings.Ne,
            dense_units = settings.forecast_dense_units,
        ) for _ in range(PositiveSample_inst.Nf)]

        # Train models
        trainer = PPC.trainers.PPC(
            encoder_model   = encoder_model,
            decoder_model   = decoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        monitor_callback = tools.callbacks.SineMonitor(
            encoder_model   = encoder_model,
            decoder_model   = decoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
            eval_iterations = 100,
            fig_dir         = save_dir
        )

        trainer.train(
            dataset                  = dataset,
            val_dataset              = val_dataset,
            forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.uniform(PositiveSample_inst.Nf), warmup=True),
            reconstruction_loss_func = PPC.losses.MeanSquaredError(),
            lambda_loss              = settings.lambda_loss,
            optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            iterations               = settings.warmup_iterations,
            callbacks                = [PPC.callbacks.TerminateOnNaN(), monitor_callback]
        )

        trainer.train(
            dataset                  = dataset,
            val_dataset              = val_dataset,
            forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.uniform(PositiveSample_inst.Nf), warmup=False),
            reconstruction_loss_func = PPC.losses.MeanSquaredError(),
            lambda_loss              = settings.lambda_loss,
            optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            callbacks                = [PPC.callbacks.TerminateOnNaN(), PPC.callbacks.RestoreOnEarlyStopping(monitor='val_forecast_loss', patience=settings.patience, models=[encoder_model, decoder_model, ar_model] + forecast_models), monitor_callback],
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