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

    def _prepare_inputs(self, input_dict, track_gradients=True):
        """Stacks all input variables into a single tensor.

        Parameters
        ----------
        input_dict : ordered dict
            The dictionary of variables that is handed to the model
            (e.g. by a dataloader).
        track_gradients : bool or list of str or list of DiffVariables
            Whether the gradients w.r.t. the inputs should be tracked.
            Tracking can be necessary for training of a PDE.
            If True, all gradients will be tracked.
            If a list of strings or variables is passed, gradient is tracked
            only for the variables in the list.
            If False, no gradients will be tracked.

        Returns
        -------
        torch.Tensor
            A single tensor containing all input variables.
        """
        # enable gradient tracking
        if isinstance(track_gradients, bool):
            if track_gradients:
                for key in input_dict:
                    input_dict[key].requires_grad = True
        elif isinstance(track_gradients, Iterable):
            for key in track_gradients:
                input_dict[key].requires_grad = True
        else:
            raise TypeError('track_gradients should be either bool or iterable.')
        # construct single torch tensor from dictionary
        return torch.cat([v for k, v in input_dict.items()], dim=1)

    def serialize(self):
        raise NotImplementedError
