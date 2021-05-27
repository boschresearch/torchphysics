"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc
import torch

from .data import Dataset


class Condition(torch.nn.Module):
    def __init__(self, name, norm, weight=1.0,
                 batch_size=1000, num_workers=0):
        self.name = name
        self.norm = norm
        self.weight = weight
        self.batch_size = batch_size
        self.num_workers = num_workers

        # variables are registered when the condition is added to a problem or variable
        self.variables = None

    @abc.abstractmethod
    def get_dataloader(self):
        """Creates and returns a dataloader for the given condition."""
        return


class DiffEqCondition(Condition):
    def __init__(self, pde, name='pde', norm=torch.nn.MSELoss(),
                 sampling_strategy='random', weight=1.0,
                 batch_size=1000, num_workers=0, dataset_size=10000):
        super().__init__(name, norm, weight,
                         batch_size=batch_size,
                         num_workers=num_workers)
        self.sampling_strategy = sampling_strategy
        self.pde = pde
        self.dataset_size = dataset_size

    def forward(self, model, data):
        u = model(data)
        err = self.pde(u, data)
        return self.norm(err, torch.zeros_like(err))

    def get_dataloader(self):
        if self.is_registered():
            dataset = Dataset(self.variables,
                              sampling_strategy=self.sampling_strategy,
                              size=self.dataset_size)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def is_registered(self):
        return self.variables is not None


class DataCondition(Condition):
    def __init__(self, data_x, data_u, name, norm=torch.nn.MSELoss(),
                 weight=1.0, batch_size=1000, num_workers=2):
        super().__init__(name, norm, weight,
                         batch_size=batch_size,
                         num_workers=num_workers)

    def forward(self, model, data):
        data, target = data
        u = model(data)
        return self.norm(u, target)

    def get_dataloader(self):
        pass


class BoundaryCondition(Condition):
    def __init__(self, name, norm, weight, batch_size, num_workers):
        super().__init__(name, norm, weight=weight, batch_size=batch_size,
                         num_workers=num_workers)
        # boundary_variable is registered when the condition is added to that variable
        self.boundary_variable = None  # string


class DirichletCondition(BoundaryCondition):
    def __init__(self, dirichlet_fun, name, norm=torch.nn.MSELoss(),
                 sampling_strategy='random', weight=1.0, batch_size=1000,
                 num_workers=0):
        super().__init__(name, norm, weight=weight, batch_size=batch_size,
                         num_workers=num_workers)
        self.dirichlet_fun = dirichlet_fun

    def forward(self, model, data):
        u = model(data)
        target = self.dirichlet_fun(data)
        return self.norm(u, target)
