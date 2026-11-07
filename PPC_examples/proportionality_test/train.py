from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.resolve() / '../../').resolve()))
import PPC
import tools
import tensorflow as tf
import numpy as np
import gc

if __name__ == '__main__':
    
    # Select GPU's
    PPC.gpu.select_gpus(
        selected_gpus = PPC.gpu.find_available(
            num_gpus        = 1,
            max_utilization = 80,
            min_free_memory = 5500,
        ),
        memory_growth = True,
    )

    # Make TensorFlow operations deterministic
    tf.config.experimental.enable_op_determinism()

    for iteration in range(100):

        # Seed per iteration, so each of the 100 repeats is reproducible but not identical
        np.random.seed(iteration)
        tf.keras.utils.set_random_seed(iteration)

        # Settings
        settings = PPC.tools.Settings(
            batch_size             = 64,
            Np                     = 1,
            Nf                     = 1,
            Ne                     = 4,
            Ng                     = 8,
            forecast_dense_units   = 16,
            lambda_loss            = 1e2,
            choice_step_one_values = [-10, 0, 10],
            normal_a               = 0.1,
            normal_b               = 2.0,
            iteration              = iteration,
            learning_rate          = 1e-4,
        )

        print(settings)

        # Select save directory
        save_dir = Path(__file__).parent.resolve() / 'models/lambda-{}/{}'.format(settings.lambda_loss, hash(settings))

        # If the model does not exist, create and train it. Otherwise, load it
        if not save_dir.exists():

            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataset instances
            step1_generator = tools.data.ChoiceStepOne(settings.choice_step_one_values)
            step2_generator = tools.data.NormalStepTwo(lambda x: settings.normal_a*x+settings.normal_b)

            dataset = tools.data.TwoStepDataset(
                batch_size      = settings.batch_size,
                step1_generator = step1_generator,
                step2_generator = step2_generator,
                seed            = 2*iteration,
            )

            val_dataset = tools.data.TwoStepDataset(
                batch_size      = settings.batch_size,
                step1_generator = step1_generator,
                step2_generator = step2_generator,
                seed            = 2*iteration+1,
            )

            # Create models
            encoder_model = tools.models.encoder_model(
                Ne = settings.Ne,
            )

            decoder_model = tools.models.decoder_model(
                Ne = settings.Ne,
            )

            ar_model = tools.models.ar_model(
                Ne = settings.Ne,
                Ng = settings.Ng,
            )

            forecast_models = [tools.models.forecast_model(
                Ng          = settings.Ng,
                Ne          = settings.Ne,
                dense_units = settings.forecast_dense_units,
            ) for _ in range(settings.Nf)]

            # Set-up parameter tracker callback
            partracker = tools.callbacks.GaussianParameterMonitor(
                encoder_model          = encoder_model,
                ar_model               = ar_model,
                forecast_models        = forecast_models,
                choice_step_one_values = settings.choice_step_one_values,
                normal_a               = settings.normal_a,
                normal_b               = settings.normal_b,
                grid_res               = 0.1,
                eval_iterations        = 500,
                fig_path               = save_dir / 'partracker.png',
            )

            # Train models
            trainer = PPC.trainers.PPC(
                encoder_model   = encoder_model,
                decoder_model   = decoder_model,
                ar_model        = ar_model,
                forecast_models = forecast_models,
            )

            trainer.train(
                dataset                  = dataset,
                val_dataset              = val_dataset,
                forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.Uniform(settings.Nf), warmup=True),
                reconstruction_loss_func = PPC.losses.MeanSquaredError(),
                lambda_loss              = settings.lambda_loss,
                optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                iterations               = 1000,
                callbacks                = [PPC.callbacks.TerminateOnNaN(), partracker]
            )
            
            trainer.train(
                dataset                  = dataset,
                val_dataset              = val_dataset,
                forecast_loss_func       = PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.Uniform(settings.Nf), warmup=False),
                reconstruction_loss_func = PPC.losses.MeanSquaredError(),
                lambda_loss              = settings.lambda_loss,
                optimizer                = tf.keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                callbacks                = [
                    PPC.callbacks.TerminateOnNaN(), 
                    PPC.callbacks.RestoreOnEarlyStopping('val_total_loss', patience=5000, models=[encoder_model, decoder_model, ar_model] + forecast_models), 
                    partracker,
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

            # Free up memory
            del dataset
            del val_dataset
            del encoder_model
            del decoder_model
            del ar_model
            del forecast_models
            del trainer

            gc.collect()
            tf.keras.backend.clear_session()