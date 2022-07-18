import torch.nn as nn

from ..model import Model
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
    """
    def __init__(self, input_space, output_space, output_neurons):
        super().__init__(input_space, output_space)
        self.output_neurons = output_neurons * output_space.dim

    def _reshape_multidimensional_output(self, output):
        if len(output.shape) == 3:
            return output.reshape(output.shape[0], output.shape[1], self.output_space.dim, 
                                  int(self.output_neurons/self.output_space.dim))
        return output.reshape(-1, self.output_space.dim, 
                              int(self.output_neurons/self.output_space.dim))



class FCTrunkNet(TrunkNet):
    """A fully connected neural networks that can be used inside a DeepONet.
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
                 hidden=(20,20,20), activations=nn.Tanh(), xavier_gains=5/3):
        super().__init__(input_space, output_space, output_neurons)

        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_space.dim, 
                                      output_dim=output_neurons, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        points = self._fix_points_order(points)
        return self._reshape_multidimensional_output(self.sequential(points))