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

        ContrastiveSamples_inst     = tools.data.ContrastiveSamples(Path(__file__).parent / 'train_dataset.h5')
        val_ContrastiveSamples_inst = tools.data.ContrastiveSamples(Path(__file__).parent / 'val_dataset.h5')

        # Create dataset instances
        dataset = tools.data.ContrastiveSamplesDataset(
            ContrastiveSamples_inst = ContrastiveSamples_inst,
            batch_size              = settings.batch_size,
            seed                    = 0,
        )

        val_dataset = tools.data.ContrastiveSamplesDataset(
            ContrastiveSamples_inst = val_ContrastiveSamples_inst,
            batch_size              = settings.batch_size,
            seed                    = 1,
        )

        # Create models
        encoder_model = tools.models.encoder_model(
            Ne      = settings.Ne,
            filters = settings.encoder_filters,
            L       = ContrastiveSamples_inst.L,
        )

        ar_model = tools.models.ar_model(
            Np = ContrastiveSamples_inst.Np,
            Ne = settings.Ne,
            Ng = settings.Ng,
        )

        forecast_models = [tools.models.forecast_model(
            Ng          = settings.Ng,
            Ne          = settings.Ne,
            dense_units = settings.forecast_dense_units,
        ) for _ in range(ContrastiveSamples_inst.Nf)]

        # Train models
        trainer = PPC.trainers.PCPC(
            encoder_model   = encoder_model,
            ar_model        = ar_model,
            forecast_models = forecast_models,
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.LogisticContrastiveLoss(PPC.weights.uniform(ContrastiveSamples_inst.Nf), warmup=True),
            optimizer          = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            iterations         = settings.warmup_iterations,
            callbacks          = [PPC.callbacks.TerminateOnNaN()],
        )

        trainer.train(
            dataset            = dataset,
            val_dataset        = val_dataset,
            forecast_loss_func = PPC.losses.LogisticContrastiveLoss(PPC.weights.uniform(ContrastiveSamples_inst.Nf), warmup=False),
            optimizer          = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
            callbacks          = [PPC.callbacks.TerminateOnNaN(), PPC.callbacks.RestoreOnEarlyStopping(monitor='val_forecast_loss', patience=settings.patience, models=[encoder_model, ar_model] + forecast_models)],
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

        ContrastiveSamples_inst.close()
        val_ContrastiveSamples_inst.close()
