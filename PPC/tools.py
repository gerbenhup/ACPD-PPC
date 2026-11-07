from typing import Tuple, Union, List
from pathlib import Path
import json
import sys
import hashlib
import argparse
import numpy as np
import tensorflow as tf

import PPC

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

        return Settings(**kwargs)

    def __str__(self):

        return str(vars(self))

    def __hash__(self):
        """Compute a stable hash of the settings, suitable for use as a model directory name

        Returns:
            int: stable hash, deterministic across interpreter runs (unlike Python's built-in string hash())
        """

        digest = hashlib.sha256(json.dumps(vars(self), sort_keys=True).encode()).hexdigest()

        return int(digest, 16) % (2**63)

class Arguments:
    """Command line arguments with default values, e.g. `Arguments(gpu=0)` accepts `--gpu 1`"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: argument names and default values. A value given on the command line is converted to the type of its default value.
        """

        argparser = argparse.ArgumentParser()

        for name in kwargs:
            argparser.add_argument('--{}'.format(name))

        cli_args = argparser.parse_args()

        for name in kwargs:
            if getattr(cli_args, name) is not None:
                setattr(self, name, type(kwargs[name])(getattr(cli_args, name)))
            else:
                setattr(self, name, kwargs[name])

def num_trainable_weights(input_: Union[tf.keras.Model, List[tf.keras.Model]]) -> int:
    """Count the number of trainable weights of a model, or of a (nested) list of models

    Args:
        input_ (Union[tf.keras.Model, List[tf.keras.Model]]): model or list of models

    Returns:
        int: number of trainable weights
    """

    if isinstance(input_, list):
        return np.sum([num_trainable_weights(model) for model in input_])
    else:
        return np.sum([np.prod(weights.shape) for weights in input_.trainable_weights])

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
        model = tf.keras.models.model_from_json(f.read(), {'PPC': sys.modules['PPC']})

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

        save_model(encoder_model, dir, 'encoder_model')
        f.write('encoder_model\n')
        save_model(decoder_model, dir, 'decoder_model')
        f.write('decoder_model\n')
        save_model(ar_model, dir, 'ar_model')
        f.write('ar_model\n')

        for i, forecast_model in enumerate(forecast_models):
            save_model(forecast_model, dir, 'forecast_model_{}'.format(i))
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
            encoder_model = load_model(dir, 'encoder_model')
        elif line == 'decoder_model':
            decoder_model = load_model(dir, 'decoder_model')
        elif line == 'ar_model':
            ar_model = load_model(dir, 'ar_model')
        elif 'forecast' in line:
            forecast_models.append(load_model(dir, line))

    return encoder_model, decoder_model, ar_model, forecast_models

def save_pcpc_models(dir: Path, encoder_model: tf.keras.Model, ar_model: tf.keras.Model, forecast_models: list[tf.keras.Model]) -> None:
    """Save all models of a PCPC pipeline (encoder, autoregressive and forecast models) to a directory

    Args:
        dir (Path): directory to save the models in
        encoder_model (tf.keras.Model): encoder model
        ar_model (tf.keras.Model): autoregressive model
        forecast_models (list[tf.keras.Model]): forecast models
    """

    dir.mkdir(parents=True, exist_ok=True)

    with open(dir / 'pcpc_models.txt', 'w') as f:

        save_model(encoder_model, dir, 'encoder_model')
        f.write('encoder_model\n')
        save_model(ar_model, dir, 'ar_model')
        f.write('ar_model\n')

        for i, forecast_model in enumerate(forecast_models):
            save_model(forecast_model, dir, 'forecast_model_{}'.format(i))
            f.write('forecast_model_{}\n'.format(i))

def load_pcpc_models(dir: Path) -> Tuple[tf.keras.Model, tf.keras.Model, list[tf.keras.Model]]:
    """Load all models of a PCPC pipeline (encoder, autoregressive and forecast models) from a directory

    Args:
        dir (Path): directory the models were saved in

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model, list[tf.keras.Model]]: encoder, autoregressive and forecast models
    """

    with open(dir / 'pcpc_models.txt', 'r') as f:
        models = f.read().strip().split('\n')

    forecast_models = []

    for line in models:
        if line == 'encoder_model':
            encoder_model = load_model(dir, 'encoder_model')
        elif line == 'ar_model':
            ar_model = load_model(dir, 'ar_model')
        elif 'forecast' in line:
            forecast_models.append(load_model(dir, line))

    return encoder_model, ar_model, forecast_models
