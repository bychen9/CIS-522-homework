from typing import List
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """constructs a scheduler"""

    def __init__(
        self,
        optimizer,
        num_batches,
        num_epochs,
        initial_learning_rate,
        eta_max,
        gamma,
        last_epoch=-1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.total_steps = num_batches * self.num_epochs
        self.eta_max = eta_max
        self.gamma = gamma
        self.last_lr = initial_learning_rate
        self.t0 = self.total_steps / 5.0
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """gets learning rate"""
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # if self.last_epoch % 300 == 0:
        #    self.last_lr = self.last_lr * self.gamma
        #    for op_params in self.optimizer.param_groups:
        #        op_params['lr'] = self.last_lr
        # -----------------------------------------
        if self.last_epoch < self.t0:
            self.last_lr = 10e-4 + self.eta_max * self.last_epoch / self.t0
        if self.last_epoch < self.total_steps:
            self.last_lr = (
                self.eta_max
                * np.cos(
                    (np.pi / 2)
                    * (self.last_epoch - self.t0)
                    / (self.total_steps - self.t0)
                )
                + 10e-6
            )
        return [self.last_lr]
        # return [i for i in self.base_lrs]
