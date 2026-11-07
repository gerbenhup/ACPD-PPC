from typing import List
import numpy as np
import tensorflow as tf

class ReduceLROnPlateau:
    """Training callback that reduces the learning rate whenever the monitored loss stagnates"""

    def __init__(self, monitor: str, patience: int, optimizer: tf.keras.optimizers.Optimizer, factor: float):
        """
        Args:
            monitor (str): name of the loss to monitor, as passed to `__call__`
            patience (int): number of iterations without improvement before reducing the learning rate
            optimizer (tf.keras.optimizers.Optimizer): optimizer whose learning rate is reduced
            factor (float): factor the learning rate is multiplied with on every reduction
        """

        self.monitor   = monitor
        self.patience  = patience
        self.optimizer = optimizer
        self.factor    = factor

        self.best_loss         = np.inf
        self.best_iteration    = 0
        self.current_iteration = 0

    def __call__(self, **kwargs) -> bool:
        """Check the monitored loss for the current iteration, reducing the learning rate if patience is exceeded

        Args:
            **kwargs: named loss values for the current iteration, must include the monitored loss name

        Returns:
            bool: always True (never interrupts training)
        """

        if self.monitor not in kwargs:
            raise ValueError('Monitor {} not provided'.format(self.monitor))

        self.current_iteration += 1

        if kwargs[self.monitor] < self.best_loss:
            self.best_loss      = kwargs[self.monitor]
            self.best_iteration = self.current_iteration

        elif self.current_iteration - self.best_iteration > self.patience:
            tf.keras.backend.set_value(self.optimizer.learning_rate, self.optimizer.learning_rate.numpy() * self.factor)
            self.best_iteration = self.current_iteration

            print()
            print('Reduced learning rate to {:.2e}'.format(self.optimizer.learning_rate.numpy()))

        return True

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

class EarlyStopping:
    """Training callback that stops training if the monitored loss stagnates, without restoring weights"""

    def __init__(self, monitor: str, patience: int):
        """
        Args:
            monitor (str): name of the loss to monitor, as passed to `__call__`
            patience (int): number of iterations without improvement before stopping training
        """

        self.monitor           = monitor
        self.patience          = patience
        self.best_loss         = np.inf
        self.best_iteration    = 0
        self.current_iteration = 0

    def __call__(self, **kwargs) -> bool:
        """Check the monitored loss for the current iteration

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
            return True

        elif self.current_iteration - self.best_iteration > self.patience:
            print()
            print('Early Stopping triggered')
            return False

class TerminateOnNaN:
    """Training callback that aborts training as soon as any reported loss becomes NaN, so no NaN model gets saved"""

    def __init__(self):
        pass

    def __call__(self, **kwargs) -> bool:
        """Check the losses reported for the current iteration for NaN values

        Args:
            **kwargs: named loss values for the current iteration

        Returns:
            bool: True if no loss is NaN

        Raises:
            FloatingPointError: if any loss is NaN
        """

        for name, value in kwargs.items():
            if np.any(np.isnan(value)):
                print()
                raise FloatingPointError('Terminate on NaN triggered by {}'.format(name))

        return True
