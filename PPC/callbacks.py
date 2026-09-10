from typing import List
import numpy as np
import tensorflow as tf

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
