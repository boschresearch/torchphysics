"""contains classes that wrap a PDE problem and
the NN model to solve this problem

classes inherit from LightningModules"""

import torch
import pytorch_lightning as pl


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
    optimizer : torch optimizer
        The PyTorch Optimizer that should be used in training
    lr : float
        The (initial) learning rate of the used optimizer. Should be set
        to 1e-3 for Adam
    metrics : WIP
        feature is not ready yet, should contain some metrics that can
        be logged in training or validation
    """

    def __init__(self, model, problem, optimizer=torch.optim.LBFGS,
                 lr=1, metrics=()):
        super().__init__()
        self.model = model
        self.problem = problem

        self.optimizer = optimizer
        self.lr = lr

        self.metrics = metrics  # does not

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def _get_dataloader(self, conditions):
        dataloader_dict = {}
        for condition in conditions:
            dataloader_dict[condition.name] = condition.get_dataloader()
        return dataloader_dict

    def train_dataloader(self):
        return self._get_dataloader(self.problem.get_train_conditions())

    def val_dataloader(self):
        # For multiple validation dataloaders, lightning need a CombinedLoader
        dataloader_dict = self._get_dataloader(self.problem.get_val_conditions())
        return pl.trainer.supporters.CombinedLoader(dataloader_dict, 'max_size_cycle')

    def _do_step(self, conditions, batch, batch_idx):
        loss = torch.Tensor(0.)
        for condition in conditions:
            data = batch[condition.name]
            c = condition(self.model, data)
            self.log(condition.name, c)
            loss += condition.weight * c
        # should be extended by custom metric logging in future
        # NOTE: we should clarify whether multiple conditions with derivatives
        #       lead to multiple computations of those derivatives,
        #       if yes, there could be more efficient solutions
        return loss

    def training_step(self, batch, batch_idx):
        return self._do_step(self.problem.get_train_conditions(), batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._do_step(self.problem.get_val_conditions(), batch, batch_idx)
