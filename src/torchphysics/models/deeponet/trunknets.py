import torch.nn as nn
import torch

from ..model import Model
from .layers import TrunkLinear
from ..fcn import _construct_FC_layers


class TrunkNet(Model):
    """A neural network that can be used inside a DeepONet-model.

    Parameters
    ----------
    input_space : Space
        The space of the points that can be put into this model.
    output_space : Space
        The number of output neurons. These neurons will only
        be used internally. The final output of the DeepONet-model will be 
        in the dimension of the output space. 
    output_neurons : int
        The number of output neurons. Will be multiplied my the dimension of the output space, 
        so each dimension will have the same number of intermediate neurons.
    trunk_input_copied : bool, optional
        If every sample function of the branch input gets evaluated at the same trunk input, 
        the evaluation process can be speed up, since the trunk only has to evaluated once
        for the whole data batch of branch inputs. 
        If this is the case, set trunk_input_copied = True.
        If for example a dataset with different trunk inputs for each branch function
        is used, set trunk_input_copied = False. Else this may lead to unexpected 
        behavior.
    """
    def __init__(self, input_space, output_space, output_neurons, 
                 trunk_input_copied=True):
        super().__init__(input_space, output_space)
        self.output_neurons = output_neurons * output_space.dim
        self.trunk_input_copied = trunk_input_copied

    def _reshape_multidimensional_output(self, output):
        if len(output.shape) == 3:
            return output.reshape(output.shape[0], output.shape[1], self.output_space.dim, 
                                  int(self.output_neurons/self.output_space.dim))
        return output.reshape(-1, self.output_space.dim, 
                              int(self.output_neurons/self.output_space.dim))

def construct_FC_trunk_layers(hidden, input_dim, output_dim, activations, xavier_gains):
    """Constructs the layer structure for a fully connected neural network.
    """
    if not isinstance(activations, (list, tuple)):
        activations = len(hidden) * [activations]
    if not isinstance(xavier_gains, (list, tuple)):
        xavier_gains = len(hidden) * [xavier_gains]

    layers = []
    layers.append(TrunkLinear(input_dim, hidden[0]))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[0])
    layers.append(activations[0])
    for i in range(len(hidden)-1):
        layers.append(TrunkLinear(hidden[i], hidden[i+1]))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[i+1])
        layers.append(activations[i+1])
    layers.append(TrunkLinear(hidden[-1], output_dim))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=1)
    return layers

class FCTrunkNet(TrunkNet):
    """A fully connected neural network that can be used inside a DeepONet.
    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points that should be
        returned by the parent DeepONet-model.
    output_neurons : int
        The number of output neurons. These neurons will only
        be used internally. The final output of the DeepONet-model will be 
        in the dimension of the output space. 
    hidden : list or tuple
        The number and size of the hidden layers of the neural network.
        The lenght of the list/tuple will be equal to the number
        of hidden layers, while the i-th entry will determine the number
        of neurons of each layer.
    activations : torch.nn or list, optional
        The activation functions of this network. 
        Deafult is nn.Tanh().
    xavier_gains : float or list, optional
        For the weight initialization a Xavier/Glorot algorithm will be used.
        Default is 5/3. 
    """
    def __init__(self, input_space, output_space, output_neurons, 
                 hidden=(20,20,20), activations=nn.Tanh(), xavier_gains=5/3, 
                 trunk_input_copied=True):
        super().__init__(input_space, output_space, output_neurons, 
                         trunk_input_copied=trunk_input_copied)

        # special layer architecture is used if trunk data is copied -> faster training
        if self.trunk_input_copied:
            layers = construct_FC_trunk_layers(hidden=hidden, input_dim=self.input_space.dim, 
                                            output_dim=self.output_neurons, 
                                            activations=activations, xavier_gains=xavier_gains)
        else:
            layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_space.dim, 
                                          output_dim=self.output_neurons, 
                                          activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        points = self._fix_points_order(points)
        return self._reshape_multidimensional_output(self.sequential(points.as_tensor))