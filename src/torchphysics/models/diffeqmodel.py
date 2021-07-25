from typing import Iterable
import torch
import torch.nn as nn


class DiffEqModel(nn.Module):
    """Neural networks that approximate the solution u of a differential equation.

    A DiffEqModel gets an ordered dict of independent variables and applies a
    Neural Network.
    """

    def __init__(self, variable_dims=None):
        super().__init__()
        self.variable_dims = variable_dims

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
        if self.variable_dims is None:
            # register the variables on which the model is trained
            self.variable_dims = {k: v.shape[1:] for k, v in input_dict.items()}
        # check whether the input has the expected variables and shape
        if self.variable_dims is None:
            print("""The correct input variables for the model have not been
                     set yet. This can lead to unexpected behaiour. Please train
                     the model or set the module.variable_dims property.""")
        try:
            # if possible, try to reorder the inputs such that they match the
            # expectation
            ordered_inputs = {}
            for k in self.variable_dims:
                if input_dict[k].shape[1:] != self.variable_dims[k]:
                    print(f"""The input {k} has the wrong dimension. This can
                              lead to unexpected behaviour.""")
                ordered_inputs[k] = input_dict[k]
            if len(ordered_inputs) != len(input_dict):
                raise KeyError
        except KeyError:
            print(f"""The model was trained on Variables with different names.
                      This can lead to unexpected behaviour.
                      Please use Variables {list(self.variable_dims.keys())}.""")
        # construct single torch tensor from dictionary
        return torch.cat([v for v in ordered_inputs.values()], dim=1)

    def serialize(self):
        raise NotImplementedError

    @property
    def output_dim(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        raise NotImplementedError
