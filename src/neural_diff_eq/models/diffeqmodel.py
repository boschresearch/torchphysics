from typing import Iterable
import torch
import torch.nn as nn


class DiffEqModel(nn.Module):
    """Neural networks that approximate the solution u of a differential equation.

    A DiffEqModel gets an ordered dict of independent variables and applies a
    Neural Network.
    """

    def __init__(self):
        super().__init__()

    def _prepare_inputs(self, input_dict):
        """Stacks all input variables into a single tensor.

        Parameters
        ----------
        input_dict : ordered dict
            The dictionary of variables that is handed to the model
            (e.g. by a dataloader).
        Returns
        -------
        torch.Tensor
            A single tensor containing all input variables.
        """
        # construct single torch tensor from dictionary
        return torch.cat([v for k, v in input_dict.items()], dim=1)

    def serialize(self):
        raise NotImplementedError
