from typing import Iterable
import torch
import numpy as np
import torch.nn as nn


class DiffEqModel(nn.Module):
    """Neural networks that approximate the solution u of a differential equation.

    A DiffEqModel gets an ordered dict of independent variables and applies a
    Neural Network.
    """

    def __init__(self, variable_dims, solution_dims, normalization_dict=None):
        super().__init__()

        self.variable_dims = variable_dims

        # compute the input dimensionality for the network
        self.input_dim = 0
        for v in self.variable_dims:
            self.input_dim += self.variable_dims[v]

        # setup normalization layer
        if normalization_dict is None:
            self.normalize = nn.Identity()
        else:
            self.normalize = nn.Linear(self.input_dim, self.input_dim)
            diag = []
            bias = []
            for k in normalization_dict:
                if isinstance(normalization_dict[k][0], Iterable):
                    diag.extend(normalization_dict[k][0])
                else:
                    diag.append(normalization_dict[k][0])
                if isinstance(normalization_dict[k][1], Iterable):
                    bias.extend(normalization_dict[k][1])
                else:
                    bias.append(normalization_dict[k][1])
            diag = 2./torch.tensor(diag)
            bias = -torch.tensor(bias)*diag
            with torch.no_grad():
                self.normalize.weight.copy_(torch.diag(diag))
                self.normalize.bias.copy_(bias)

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
            print(f"""The given variable names do not fit variable_dims.
                      This can lead to unexpected behaviour.
                      Please use Variables {list(self.variable_dims.keys())}.""")
        # construct single torch tensor from dictionary
        x = torch.cat([v for v in ordered_inputs.values()], dim=1)
        return self.normalize(x)

    def _prepare_outputs(self, y):
        """Divides the model output to the given solution dimensions.

        Parameters
        ----------
        y : torch.Tensor
            The output tensor, after evaluating the input data.
        Returns
        -------
        ordered dict
            A dictionary containing the output tensors in form and order 
            of solution_dims. 
        """
        idx = 0
        dct = {}
        for s in self.solution_dims:
            dct[s] = y[:, idx:idx+self.solution_dims[s]]
            idx += self.solution_dims[s]
        return dct

    def get_layers(self):
        """Returns the layers structure of the model.

        Returns
        -------
        torch.nn.ModuleList
            A list containg the used layers, in the correct order.
        """
        return self.layers

    def set_weights_of_layer(self, new_weights, layer):
        """Changes the weights of the specified layer to the given values.

        Parameters
        ----------
        new_weights : number, list, array or tensor
            The value of the new weights. If a single number is given as an input,
            all weights of the layer will be set to this number.
            If a list, array or tensor is given, they have to be of the right
            input and output dimension w.r.t the layer they replace. 
        layer : int
            The index of the layer that should be changed. To get the structure of 
            the used model, one can call .get_layers().
        """
        if isinstance(new_weights, (int, float)):
            self.layers[layer].weight.data.fill_(float(new_weights))
        else:
            new_weights = self._change_to_tensor(new_weights)
            if not new_weights.shape == self.layers[layer].weight.shape:
                raise ValueError(f"""The shape of new_weight: {new_weights.shape} 
                                     does not fit the shape of the layer: 
                                     {self.layers[layer].weight.shape}.""")
            self.layers[layer].weight.data = torch.nn.Parameter(new_weights)
 
    def set_biases_of_layer(self, new_biases, layer):
        """Changes the biases of the specified layer to the given values.

        Parameters
        ----------
        new_biases : number, list, array or tensor
            The value of the new biases. If a single number is given as an input,
            all biases of the layer will be set to this number.
            If a list, array or tensor is given, they have to be of the right
            dimension w.r.t the bias they replace. 
        layer : int
            The index of the layer that should be changed. To get the structure of 
            the used model, one can call .get_layers().
        """
        if isinstance(new_biases, (int, float)):
            self.layers[layer].bias.data.fill_(float(new_biases))
        else:
            new_biases = self._change_to_tensor(new_biases)
            if not new_biases.shape == self.layers[layer].bias.shape:
                raise ValueError(f"""The shape of new_weight: {new_biases.shape} 
                                     does not fit the shape of the layer: 
                                     {self.layers[layer].bias.shape}.""")
            self.layers[layer].bias.data = torch.nn.Parameter(new_biases)

    def set_activation_function_of_layer(self, new_func, layer):
        """Changes the biases of the specified layer to the given values.

        Parameters
        ----------
        new_func : function handle
            The new activation function of the layer. 
            If this is a custom function one has to implement a forward and
            backward call.
        layer : int
            The index of the layer that should be changed. To get the structure of 
            the used model, one can call .get_layers().
        """
        self.layers[layer] = new_func

    def _change_to_tensor(self, input):
        # Changes new weights and biases to torch.tensors
        if isinstance(input, torch.Tensor):
            return input
        elif isinstance(input, np.ndarray):
            return torch.from_numpy(input)
        elif isinstance(input, (tuple, list)):
            return torch.FloatTensor(input)
        else:
            raise ValueError(f"""Expected the new weights/biases to be
                                 number, list, np.array or tensor, not
                                 {type(input)}""")

    def serialize(self):
        raise NotImplementedError
