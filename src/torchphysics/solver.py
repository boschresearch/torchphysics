"""contains classes that wrap a PDE problem and
the NN model to solve this problem

classes inherit from LightningModules"""

import json
from typing import Dict

import torch
import pytorch_lightning as pl
from .utils.plot import _scatter


class PINNModule(pl.LightningModule):
    """A LightningModule to solve PDEs using the PINN approach

    Parameters
    ----------
    model : models.DiffEqModule object
        A PyTorch Module that should inherit from DiffEqModule. This
        Neural Network is trained to approximate to solution of the
        given problem
    problem : problem object
        A problem object that includes the DE and its whole Setting,
        i.e. variables with their domains and boundary conditions
    optimizer : torch optimizer class
        The PyTorch Optimizer that should be used in training
    optim_params : dic
        Additional parameters for the optimizer
    lr : float
        The (initial) learning rate of the used optimizer. Should be set
        to 1e-3 for Adam
    log_plotter : Plotter
        A plotter from utils.plot, that plots the solution at desired
        training epochs to the tensorboard
    scheduler : torch.optim.lr_scheduler
        A scheduler to change/adjust the learning rate based on the number of epochs
        or loss size
    """

    def __init__(self, model, optimizer=torch.optim.LBFGS,
                 lr=1, optim_params={}, log_plotter=None,
                 scheduler=None):
        super().__init__()
        self.model = model

        self.optimizer = optimizer
        self.lr = lr
        self.optim_params = optim_params
        self.scheduler = scheduler

        self.log_plotter = log_plotter
        self.variable_dims = None

    def to(self, device):
        if self.trainer is not None:
            self.trainer.datamodule.parameters.to(device)
        return super().to(device)

    def serialize(self):
        dct = {}
        dct['name'] = 'PINNModule'
        dct['model'] = self.model.serialize()
        if self.trainer is not None:
            dct['problem'] = self.trainer.datamodule.serialize()
        else:
            dct['problem'] = None
        dct['optimizer'] = {'name': self.optimizer.__name__,
                            'lr': self.lr
                            }
        dct['optim_params'] = self.optim_params
        return dct

    def forward(self, inputs):
        """
        Run the model on a given input batch, without tracking gradients.
        """
        assert isinstance(inputs, Dict), "Please pass a dict of variables and data."
        return self.model.forward(inputs)

    @property
    def output_dim(self):
        return self.model.output_dim

    @property
    def input_dim(self):
        return self.model.input_dim

    def on_train_start(self):
        # log summary to tensorboard
        if self.logger is not None:
            self.logger.experiment.add_text(
                tag='summary',
                text_string=json.dumps(
                    self.serialize(),
                    indent='&emsp; &emsp;').replace('\n', '  \n')
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(
            list(self.model.parameters()) +
            list(self.trainer.datamodule.parameters.values()),
            lr=self.lr,
            **self.optim_params)
        if self.scheduler is None:
            return optimizer
        lr_scheduler = self.scheduler['class'](
            optimizer, **self.scheduler['args'])
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # maybe this slows down training a bit
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        conditions = self.trainer.datamodule.get_train_conditions()
        for name in conditions:
            data = batch[name]
            # log scatter plots of the used training data
            # self.log_condition_data_plot(name, conditions[name], data)
            # get error for this conditions
            c = conditions[name](self.model, data)
            self.log(f'{name}/train', c)
            # accumulate weighted error
            loss = loss + conditions[name].weight * c
        self.log('loss/train', loss)
        if self.log_plotter is not None:
            self.log_plot()
        for pname, p in self.trainer.datamodule.parameters.items():
            if p.shape == [1]:
                self.log(pname, p.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=self.device)
        conditions = self.trainer.datamodule.get_val_conditions()
        for name in conditions:
            # if a condition does not require input gradients, we do not
            # compute them during validation
            torch.set_grad_enabled(conditions[name].track_gradients is not False)
            data = batch[name]
            c = conditions[name](self.model, data)
            self.log(f'{name}/val', c)
            loss = loss + conditions[name].weight * c
        self.log('loss/val', loss)

    def log_condition_data_plot(self, name, condition, data):
        if self.global_step % 10 == 0 and self.logger is not None:
            if condition.get_data_plot_variables() is not None:
                fig = _scatter(plot_variables=condition.get_data_plot_variables(),
                               data=data)
                self.logger.experiment.add_figure(tag=name+'_data',
                                                  figure=fig,
                                                  global_step=self.global_step)

    def log_plot(self):
        if self.global_step % self.log_plotter.log_interval == 0 \
             and self.logger is not None:
            fig = self.log_plotter.plot(model=self.model,
                                        device=self.device)
            self.logger.experiment.add_figure(tag='plot',
                                              figure=fig,
                                              global_step=self.global_step)
