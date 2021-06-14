"""contains classes that wrap a PDE problem and
the NN model to solve this problem

classes inherit from LightningModules"""

import json

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
    lr : float
        The (initial) learning rate of the used optimizer. Should be set
        to 1e-3 for Adam
    """

    def __init__(self, model, problem, optimizer=torch.optim.LBFGS,
                 lr=1, log_plotter=None):
        super().__init__()
        self.model = model
        self.problem = problem

        self.optimizer = optimizer
        self.lr = lr

        self.log_plotter = log_plotter

    def serialize(self):
        dct = {}
        dct['name'] = 'PINNModule'
        dct['model'] = self.model.serialize()
        dct['problem'] = self.problem.serialize()
        dct['optimizer'] = {'name': self.optimizer.__class__.__name__,
                            'lr': self.lr
                            }
        return dct

    def forward(self, inputs):
        """Run the model on a given input batch, without tracking gradients
        """
        return self.model.forward(inputs)

    def on_train_start(self):
        self.logger.experiment.add_text(
            tag='summary',
            text_string=json.dumps(
                self.serialize(),
                indent='&emsp; &emsp;').replace('\n', '  \n')
        )

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def _get_dataloader(self, conditions):
        dataloader_dict = {}
        for name in conditions:
            dataloader_dict[name] = conditions[name].get_dataloader()
        return dataloader_dict

    def train_dataloader(self):
        if self.problem.get_train_conditions() == {}:
            return None
        return self._get_dataloader(self.problem.get_train_conditions())

    def val_dataloader(self):
        # For multiple validation dataloaders, lightning needs a CombinedLoader
        if self.problem.get_val_conditions() == {}:
            return None
        dataloader_dict = self._get_dataloader(self.problem.get_val_conditions())
        return pl.trainer.supporters.CombinedLoader(dataloader_dict, 'max_size_cycle')

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        conditions = self.problem.get_train_conditions()
        for name in conditions:
            data = batch[name]
            # log scatter plots of the used training data
            self.log_condition_data_plot(name, conditions[name], data)
            # get error for this conditions
            c = conditions[name](self.model, data)
            self.log(name, c)
            # accumulate weighted error
            loss = loss + conditions[name].weight * c
        if self.log_plotter is not None:
            self.log_plot()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=self.device)
        conditions = self.problem.get_val_conditions()
        for name in conditions:
            # if a condition does not require input gradients, we do not
            # compute them during validation
            torch.set_grad_enabled(conditions[name].requires_input_grad)
            data = batch[name]
            c = conditions[name](self.model, data)
            self.log(name, c)
            loss = loss + conditions[name].weight * c

    def log_condition_data_plot(self, name, condition, data):
        if self.global_step % 10 == 0:
            if condition.get_data_plot_variables() is not None:
                fig = _scatter(plot_variables=condition.get_data_plot_variables(),
                               data=data)
                self.logger.experiment.add_figure(tag=name+'_data',
                                                  figure=fig,
                                                  global_step=self.global_step)

    def log_plot(self):
        if self.global_step % self.log_plotter.log_interval == 0:
            fig = self.log_plotter.plot(model=self.model,
                                        device=self.device)
            self.logger.experiment.add_figure(tag='plot',
                                              figure=fig,
                                              global_step=self.global_step)
