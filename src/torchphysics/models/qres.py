import torch
import torch.nn as nn

from .model import Model
from ..problem.spaces import Points


class Quadratic(nn.Module):
    """Implements a quadratic layer of the form:  W_1*x (*) W_2*x + W_1*x + b.
    Here (*) means the hadamard product of two vectors (elementwise multiplication).
    W_1, W_2 are weight matrices and b is a bias vector.

    Parameters
    ----------
    in_features : int 
        size of each input sample.
    out_features :
        size of each output sample.
    xavier_gains : float or list
        For the weight initialization a Xavier/Glorot algorithm will be used.
        The gain can be specified over this value.
        Default is 5/3. 
    """
    def __init__(self, in_features, out_features, xavier_gains):
        super().__init__()
        bias = torch.nn.init.xavier_normal_(torch.zeros(1, out_features), 
                                            gain=xavier_gains) 
        self.bias = torch.nn.Parameter(bias)
        self.linear_weights = torch.nn.Linear(in_features=in_features, 
                                              out_features=out_features, 
                                              bias=False)
        torch.nn.init.xavier_normal_(self.linear_weights.weight, gain=xavier_gains)    
        self.quadratic_weights = torch.nn.Linear(in_features=in_features, 
                                                 out_features=out_features,
                                                 bias=False)
        torch.nn.init.xavier_normal_(self.quadratic_weights.weight, gain=xavier_gains)                                 

    def forward(self, points):
        linear_out = self.linear_weights(points)
        quad_out = self.quadratic_weights(points)
        return quad_out * linear_out + linear_out + self.bias

    @property
    def in_features(self):
        return self.linear_weights.weight.shape[1]

    @property
    def out_features(self):
        return self.linear_weights.weight.shape[0]


class QRES(Model):
    """Implements the quadratic residual networks from [1].
    Instead of a linear layer, a quadratic layer W_1*x (*) W_2*x + W_1*x + b
    will be used. Here (*) means the hadamard product of two vectors 
    (elementwise multiplication).

    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    hidden : list or tuple
        The number and size of the hidden layers of the neural network.
        The lenght of the list/tuple will be equal to the number
        of hidden layers, while the i-th entry will determine the number
        of neurons of each layer.
        E.g hidden = (10, 5) -> 2 layers, with 10 and 5 neurons.
    activations : torch.nn or list, optional
        The activation functions of this network. If a single function is passed
        as an input, will use this function for each layer.
        If a list is used, will use the i-th entry for i-th layer.
        Deafult is nn.Tanh().
    xavier_gains : float or list, optional
        For the weight initialization a Xavier/Glorot algorithm will be used.
        The gain can be specified over this value.
        Default is 5/3. 

    Notes
    -----
    ..  [1] Jie Bu and Anuj Karpatne, "Quadratic Residual Networks: 
        A New Class of Neural Networks for Solving Forward and Inverse Problems 
        in Physics Involving PDEs", 2021
    """
    def __init__(self,
                 input_space,
                 output_space,
                 hidden=(20,20,20),
                 activations=nn.Tanh(),
                 xavier_gains=5/3):
        super().__init__(input_space, output_space)

        if not isinstance(activations, (list, tuple)):
            activations = len(hidden) * [activations]
        if not isinstance(xavier_gains, (list, tuple)):
            xavier_gains = len(hidden) * [xavier_gains]

        layers = []
        layers.append(Quadratic(self.input_space.dim, hidden[0], xavier_gains[0]))
        layers.append(activations[0])
        for i in range(len(hidden)-1):
            layers.append(Quadratic(hidden[i], hidden[i+1], xavier_gains[i+1]))
            layers.append(activations[i+1])
        layers.append(Quadratic(hidden[-1], self.output_space.dim, 1.0))

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        return Points(self.sequential(points), self.output_space)