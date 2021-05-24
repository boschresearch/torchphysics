import abc
import torch


class DiffEqCondition(torch.nn.Module):
    def __init__(self, name, norm, weight=1.0):
        self.name = name
        self.norm = norm
        self.weight = weight

    @abc.abstractmethod
    def get_dataloader():
        """Creates and returns a dataloader for the given condition."""
        return


class PDECondition(DiffEqCondition):
    def __init__(self, pde, name, norm, weight=1.0):
        super().__init__(name, norm, weight)
        self.pde = pde

    def forward(self, model, data):
        u = model(data)
        err = self.pde(u, data)
        return self.norm(err, torch.zeros_like(err))


class DataCondition(DiffEqCondition):
    def __init__(self, name, norm, weight=1.0):
        super().__init__(name, norm, weight)

    def forward(self, model, data):
        data, target = data
        u = model(data)
        return self.norm(u, target)
