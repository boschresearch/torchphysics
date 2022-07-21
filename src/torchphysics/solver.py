from typing import Dict
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl


class OptimizerSetting:
    """
    A helper class to sum up the optimization setup in a single class.
    """
    def __init__(self, optimizer_class, lr, optimizer_args={}, scheduler_class=None,
                 scheduler_args={}, scheduler_frequency=1):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.optimizer_args = optimizer_args
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args
        self.scheduler_frequency = scheduler_frequency


class Solver(pl.LightningModule):
    """
    A LightningModule that handles optimization and metric logging of given
    conditions.

    Parameters
    ----------
    train_conditions : tuple or list
        Tuple or list of conditions to be optimized. The weighted sum of their
        losses will be computed and minimized.
    val_conditions : tuple or list
        Conditions to be tracked during the validation part of the training, can
        be used e.g. to track errors comparede to measured data.
    optimizer_setting : OptimizerSetting
        A OptimizerSetting object that contains all necessary parameters for
        optimizing, see :class:`OptimizerSetting`.
    """
    def __init__(self,
                 train_conditions,
                 val_conditions=(),
                 optimizer_setting=OptimizerSetting(torch.optim.Adam,
                                                    1e-3)):
        super().__init__()
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.optimizer_setting = optimizer_setting
    
    def train_dataloader(self):
        # HACK: create an empty trivial dataloader, since real data is loaded
        # in conditions
        steps = self.trainer.max_steps
        if steps is None:
            warnings.warn("The maximum amount of iterations should be defined in"
                "trainer.max_steps. If undefined, the solver will train in epochs"
                "of 1000 steps.")
            steps = 1000
        return torch.utils.data.DataLoader(torch.empty(steps))
    
    def val_dataloader(self):
        # HACK: we perform only a single step during validation,
        return torch.utils.data.DataLoader(torch.empty(1))

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler = self.scheduler['class'](optimizer, **self.scheduler['args'])
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate', 
                        'interval': 'epoch', 'frequency': 1}
        for input_name in self.scheduler:
            if not input_name in ['class', 'args']:
                lr_scheduler[input_name] = self.scheduler[input_name]
        return lr_scheduler

    def on_train_start(self):
        # move static data to correct device:
        for condition in self.train_conditions:
            condition._move_static_data(self.device)
        for condition in self.val_conditions:
            condition._move_static_data(self.device)
        self.n_training_step = 0

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        for condition in self.train_conditions:
            cond_loss = condition.weight * condition(device=self.device, iteration=self.n_training_step)
            self.log(f'train/{condition.name}', cond_loss)
            loss = loss + cond_loss

        self.log('train/loss', loss)
        self.n_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        for condition in self.val_conditions:
            torch.set_grad_enabled(condition.track_gradients is not False)
            self.log(f'val/{condition.name}', condition.weight * condition(device=self.device))

    def configure_optimizers(self):
        optimizer = self.optimizer_setting.optimizer_class(
            self.parameters(),
            lr = self.optimizer_setting.lr,
            **self.optimizer_setting.optimizer_args
        )
        if self.optimizer_setting.scheduler_class is None:
            return optimizer

        lr_scheduler = self.optimizer_setting.scheduler_class(optimizer,
            **self.optimizer_setting.scheduler_args
        )
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate', 
                        'interval': 'step',
                        'frequency': self.optimizer_setting.scheduler_frequency}
        for input_name in self.optimizer_setting.scheduler_args:
            lr_scheduler[input_name] = self.optimizer_setting.scheduler_args[input_name]
        return [optimizer], [lr_scheduler]
