import torch
import torch.nn as nn
import math

from .model import Model
from ..problem.spaces import Points


def _construct_FC_layers(hidden, input_dim, output_dim, activations, xavier_gains):
    """Constructs the layer structure for a fully connected neural network.
    """
    if not isinstance(activations, (list, tuple)):
        activations = len(hidden) * [activations]
    if not isinstance(xavier_gains, (list, tuple)):
        xavier_gains = len(hidden) * [xavier_gains]

    layers = []
    layers.append(nn.Linear(input_dim, hidden[0]))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[0])
    layers.append(activations[0])
    for i in range(len(hidden)-1):
        layers.append(nn.Linear(hidden[i], hidden[i+1]))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[i+1])
        layers.append(activations[i+1])
    layers.append(nn.Linear(hidden[-1], output_dim))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=1)
    return layers


class FCN(Model):
    """A simple fully connected neural network.

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
    """
    def __init__(self,
                 input_space,
                 output_space,
                 hidden=(20,20,20),
                 activations=nn.Tanh(),
                 xavier_gains=5/3):
        super().__init__(input_space, output_space)

        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_space.dim, 
                                      output_dim=self.output_space.dim, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        points = self._fix_points_order(points)
        return Points(self.sequential(points), self.output_space)


class Harmonic_FCN(Model):
    """A fully connected neural network, that for the input :math:`x` will also 
    compute (and use) the values 
    :math:`(\cos(\pi x), \sin(\pi x), ..., \cos(n \pi x), \sin(n \pi x))`.
    as an input. See for example [1], for some theoretical background, on why this may be 
    advantageous.
    Should be used in sequence with a normalization layer, to get inputs in the range
    of [-1, 1] with the cos/sin functions.

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
    max_frequenz : int
        The highest frequenz that should be used in the input computation.
        Equal to :math:`n` in the above describtion.
    min_frequenz : int
        The smallest frequenz that should be used. Usefull, if it is expected, that 
        only higher frequenzies appear in the solution.
        Default is 0.
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
    ..  [1] Tancik, Matthew and Srinivasan, Pratul P. and Mildenhall, Ben et al., 
        "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional 
        Domains", 2020
    """
    def __init__(self, input_space, output_space, max_frequenz : int, 
                 hidden=(20,20,20), min_frequenz : int = 0,
                 activations=nn.Tanh(), xavier_gains=5/3):
        assert max_frequenz > min_frequenz, "used max frequenz has to be > min frequenz"
        super().__init__(input_space, output_space)
        self.max_frequenz = max_frequenz
        self.min_frequenz = min_frequenz
        layers = _construct_FC_layers(hidden=hidden, 
                                      input_dim=(2*(max_frequenz-min_frequenz)+1) * self.input_space.dim, 
                                      output_dim=self.output_space.dim, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        points = self._fix_points_order(points).as_tensor
        points_list = [points]
        for i in range(self.min_frequenz, self.max_frequenz):
            points_list.append(torch.cos((i+1) * math.pi * points))
            points_list.append(torch.sin((i+1) * math.pi * points))
        points = torch.cat(points_list, dim=-1)
        return Points(self.sequential(points), self.output_space)