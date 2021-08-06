import torch
import torch.nn as nn


class DiffEqModel(nn.Module):
    """Neural networks that approximate the solution u of a differential equation.

    A DiffEqModel gets an ordered dict of independent variables and applies a
    Neural Network.
    """

    def __init__(self, variable_dims, solution_dims):
        super().__init__()

        self.variable_dims = variable_dims
        # compute the input dimensionality for the network

        self.input_dim = 0
        for v in self.variable_dims:
            self.input_dim += self.variable_dims[v]

        # compute output dimensionality
        self.solution_dims = solution_dims

        self.output_dim = 0
        for s in self.solution_dims:
            self.output_dim += self.solution_dims[s]

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
        try:
            # if possible, try to reorder the inputs such that they match the
            # expectation
            ordered_inputs = {}
            for k in self.variable_dims:
                if input_dict[k].shape[-1] != self.variable_dims[k] \
                     or len(input_dict[k].shape) > 2:
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

    def _prepare_outputs(self, y):
        idx = 0
        dct = {}
        for s in self.solution_dims:
            dct[s] = y[:, idx:idx+self.solution_dims[s]]
            idx += self.solution_dims[s]
        return dct

    def serialize(self):
        raise NotImplementedError
