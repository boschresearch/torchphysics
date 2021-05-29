"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc
import torch

from .data import Dataset, DataDataset


class Condition(torch.nn.Module):
    def __init__(self, name, norm, weight=1.0,
                 batch_size=1000, num_workers=0,
                 requires_input_grad=True):
        super().__init__()
        self.name = name
        self.norm = norm
        self.weight = weight
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.requires_input_grad = requires_input_grad

        # variables are registered when the condition is added to a problem or variable
        self.variables = None

    @abc.abstractmethod
    def get_dataloader(self):
        """Creates and returns a dataloader for the given condition."""
        return

    def is_registered(self):
        return self.variables is not None


class DiffEqCondition(Condition):
    def __init__(self, pde, norm, name='pde',
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


class DataCondition(Condition):
    def __init__(self, data_x, data_u, name, norm,
                 weight=1.0, batch_size=1000, num_workers=2):
        super().__init__(name, norm, weight,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         requires_input_grad=False)
        self.data_x = data_x
        self.data_u = data_u

    def forward(self, model, data):
        data, target = data
        u = model(data)
        return self.norm(u, target)

    def get_dataloader(self):
        if self.is_registered():
            dataset = DataDataset(self.variables,
                                  data_x=self.data_x,
                                  data_u=self.data_u)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")


class BoundaryCondition(Condition):
    def __init__(self, name, norm, weight, batch_size, num_workers,
                 requires_input_grad, boundary_sampling_strategy):
        super().__init__(name, norm, weight=weight, batch_size=batch_size,
                         num_workers=num_workers,
                         requires_input_grad=requires_input_grad)
        # boundary_variable is registered when the condition is added to that variable
        self.boundary_variable = None  # string
        self.boundary_sampling_strategy = boundary_sampling_strategy


class DirichletCondition(BoundaryCondition):
    def __init__(self, dirichlet_fun, name, norm,
                 sampling_strategy='random', boundary_sampling_strategy='random',
                 weight=1.0, batch_size=1000, num_workers=0, dataset_size=10000):
        super().__init__(name, norm, weight=weight, batch_size=batch_size,
                         num_workers=num_workers, requires_input_grad=False,
                         boundary_sampling_strategy=boundary_sampling_strategy)
        self.dirichlet_fun = dirichlet_fun
        self.sampling_strategy = sampling_strategy
        self.dataset_size = dataset_size

    def forward(self, model, data):
        u = model(data)
        target = self.dirichlet_fun(data)
        return self.norm(u, target)

    def get_dataloader(self):
        if self.is_registered():
            dataset = Dataset(self.variables,
                              sampling_strategy=self.sampling_strategy,
                              boundary_sampling_strategy=self.boundary_sampling_strategy,
                              size=self.dataset_size,
                              boundary=self.boundary_variable)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")
