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
                 optim_params={}, lr=1, log_plotter=None,
                 scheduler=None):
        super().__init__()
        self.model = model
        self.datamodule = problem

        self.optimizer = optimizer
        self.lr = lr
        self.optim_params = optim_params
        self.scheduler = scheduler

        self.log_plotter = log_plotter

    def serialize(self):
        dct = {}
        dct['name'] = 'PINNModule'
        dct['model'] = self.model.serialize()
        dct['problem'] = self.datamodule.serialize()
        dct['optimizer'] = {'name': self.optimizer.__class__.__name__,
                            'lr': self.lr
                            }
        dct['optim_params'] = self.optim_params
        return dct

    def forward(self, inputs):
        """Run the model on a given input batch, without tracking gradients
        """
        try:
            ordered_inputs = {k: inputs[k] for k in self.datamodule.variables.keys()}
            if len(ordered_inputs) < len(inputs):
                raise KeyError
        except KeyError:
            print(f"""The model was trained on Variables with different names.
                      Please use keys {self.datamodule.variables.keys()}.""")
        return self.model.forward(ordered_inputs)

    def on_train_start(self):
        self.logger.experiment.add_text(
            tag='summary',
            text_string=json.dumps(
                self.serialize(),
                indent='&emsp; &emsp;').replace('\n', '  \n')
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   **self.optim_params)
        if self.scheduler is None:
            return optimizer
        lr_scheduler = self.scheduler['class'](
            optimizer, **self.scheduler['args'])
        return [optimizer], [lr_scheduler]

    def _get_dataloader(self, conditions):
        dataloader_dict = {}
        for name in conditions:
            dataloader_dict[name] = conditions[name].get_dataloader()
        return dataloader_dict

    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=self.device, requires_grad=True)
        conditions = self.datamodule.get_train_conditions()
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
        return loss

    def validation_step(self, batch, batch_idx):
        loss = torch.zeros(1, device=self.device)
        conditions = self.datamodule.get_val_conditions()
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
