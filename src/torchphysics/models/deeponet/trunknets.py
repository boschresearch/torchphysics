import torch.nn as nn
import torch

from ..model import Model
from .layers import TrunkLinear
from ..fcn import _construct_FC_layers
from ...problem.spaces.points import Points

class TrunkNet(Model):
    """A neural network that can be used inside a DeepONet-model.

    Parameters
    ----------
    input_space : Space
        The space of the points that can be put into this model.
    default_trunk_input : tp.spaces.Points, torch.tensor
        The default input for the trunk net if no other values are specified.
        This default is used when the DeepONet only should be evaluated for new
        branch inputs, but the trunk input stays fixed. Shape should be of the form
        (N_batch, ..., dimension of input space), here ... can be abritrary many dimensions.
        If trunk_input_copied=True, N_batch needs to be equal to 1.
    trunk_input_copied : bool, optional
        If every sample function of the branch input gets evaluated at the same trunk input,
        the evaluation process can be speed up, since the trunk only has to evaluated once
        for the whole data batch of branch inputs.
        If this is the case, set trunk_input_copied = True.
        If for example a dataset with different trunk inputs for each branch function
        is used, set trunk_input_copied = False. Else this may lead to unexpected
        behavior.
    """

    def __init__(self, input_space, default_trunk_input, trunk_input_copied=True):
        super().__init__(input_space, output_space=None)
        self.output_neurons = 0
        self.trunk_input_copied = trunk_input_copied

        if torch.is_tensor(default_trunk_input):
            self.default_trunk_input = Points(default_trunk_input, input_space)
        elif isinstance(default_trunk_input, Points):
            self.default_trunk_input = default_trunk_input
        else:
            raise ValueError("Provided default input is not supported!")
        
        self.default_trunk_input = self._fix_points_order(self.default_trunk_input)

    def finalize(self, output_space, output_neurons):
        """Method to set the output space and output neurons of the network.
        Will be called once the BranchNet is connected to the TrunkNet, so
        that both will have a fitting output shape.

        Parameters
        ----------
        output_space : Space
            The space in which the final output of the DeepONet will belong to.
        output_neurons : int
            The number of output neurons. Will be multiplied my the dimension of the
            output space, so each dimension will have the same number of
            intermediate neurons.

        """
        self.output_neurons = output_neurons
        self.output_space = output_space

    def _reshape_multidimensional_output(self, output):
        return output.reshape(
            *output.shape[:-1], self.output_space.dim, self.output_neurons
        )


def construct_FC_trunk_layers(hidden, input_dim, output_dim, activations, xavier_gains):
    """Constructs the layer structure for a fully connected neural network."""
    if not isinstance(activations, (list, tuple)):
        activations = len(hidden) * [activations]
    if not isinstance(xavier_gains, (list, tuple)):
        xavier_gains = len(hidden) * [xavier_gains]

    layers = []
    layers.append(TrunkLinear(input_dim, hidden[0]))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[0])
    layers.append(activations[0])
    for i in range(len(hidden) - 1):
        layers.append(TrunkLinear(hidden[i], hidden[i + 1]))
        torch.nn.init.xavier_normal_(layers[-1].weight, gain=xavier_gains[i + 1])
        layers.append(activations[i + 1])
    layers.append(TrunkLinear(hidden[-1], output_dim))
    torch.nn.init.xavier_normal_(layers[-1].weight, gain=1)
    return layers


class FCTrunkNet(TrunkNet):
    """A fully connected neural network that can be used inside a DeepONet.
    Parameters
    ----------
    input_space : Space
        The space of the points the can be put into this model.
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

    def __init__(
        self,
        input_space,
        default_trunk_input,
        hidden=(20, 20, 20),
        activations=nn.Tanh(),
        xavier_gains=5 / 3,
        trunk_input_copied=True,
    ):
        super().__init__(input_space, default_trunk_input,trunk_input_copied=trunk_input_copied)
        self.hidden = hidden
        self.activations = activations
        self.xavier_gains = xavier_gains

    def finalize(self, output_space, output_neurons):
        super().finalize(output_space, output_neurons)
        # special layer architecture is used if trunk data is copied -> faster training
        if self.trunk_input_copied:
            layers = construct_FC_trunk_layers(
                hidden=self.hidden,
                input_dim=self.input_space.dim,
                output_dim=self.output_neurons*self.output_space.dim,
                activations=self.activations,
                xavier_gains=self.xavier_gains,
            )
        else:
            layers = _construct_FC_layers(
                hidden=self.hidden,
                input_dim=self.input_space.dim,
                output_dim=self.output_neurons*self.output_space.dim,
                activations=self.activations,
                xavier_gains=self.xavier_gains,
            )

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        if points:
            points = self._fix_points_order(points)
            return self._reshape_multidimensional_output(self.sequential(points.as_tensor))
        else:
            current_device = next(self.sequential[0].parameters()).device
            self.default_trunk_input = self.default_trunk_input.to(current_device)
            return self._reshape_multidimensional_output(
                    self.sequential(self.default_trunk_input.as_tensor)
                )
