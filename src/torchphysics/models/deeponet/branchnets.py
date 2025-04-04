import torch
import abc
import torch.nn as nn

from ..model import Model
from ..fcn import _construct_FC_layers
from ...problem.domains.functionsets.functionset import FunctionSet
from ...utils.user_fun import UserFunction
from ...problem.spaces.points import Points


class BranchNet(Model):
    """A neural network that can be used inside a DeepONet-model.

    Parameters
    ----------
    function_space : Space
        The space of functions that can be put in this network.
    grid : torchphysics.spaces.Points
        The points at which the input functions should
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of grid points.

    """

    def __init__(self, function_space, grid):
        super().__init__(function_space, output_space=None)
        # Transform to points to have unified checks
        if torch.is_tensor(grid):
            grid = Points(grid, function_space.input_space)

        self.output_neurons = 0
        self.register_buffer("grid_buffer", grid.as_tensor)
        self.current_out = torch.empty(0)

    def __getattr__(self, name):
        if name == "grid":
            return object.__getattribute__(self, "grid")
        # Call parent __getattr__ for other cases
        return super().__getattr__(name)

    @property
    def grid(self):
        return Points(self.grid_buffer, self.input_space.input_space)

    def finalize(self, output_space, output_neurons):
        """Method to set the output space and output neurons of the network.
        Will be called once the BranchNet is connected to the TrunkNet, so
        that both will have a fitting output shape.

        output_space : Space
            The space to which the final output of the DeepONet will belong to.
        output_neurons : int
            The number of output neurons. Will be multiplied my the dimension of the
            output space, so each dimension will have the same amount of
            intermediate neurons.
        """
        self.output_neurons = output_neurons
        self.output_space = output_space

    def _reshape_multidimensional_output(self, output):
        return output.reshape(
            -1, self.output_space.dim, int(self.output_neurons / self.output_space.dim)
        )

    @abc.abstractmethod
    def forward(self, discrete_function_batch, device="cpu"):
        """Evaluated the network at a given function batch. Should not be called
        directly, rather use the method ``.fix_input``.

        Parameters
        ----------
        discrete_function_batch : tp.space.Points
            The points object of discrete function values to evaluate the model.
        device : str, optional
            The device where the data lays. Default is 'cpu'.

        Notes
        -----
        Will, in general, not return anything. The output of the network will be saved
        internally to be used multiple times.
        """
        raise NotImplementedError

    def fix_input(self, function, device="cpu"):
        """Fixes the branch net for a given function. The branch net will
        be evaluated for the given function and the output saved in ``current_out``.

        Parameters
        ----------
        function : callable, torchphysics.domains.FunctionSet, torch.Tensor,
                    torchphysics.spaces.Points
            The function(s) for which the network should be evaluaded.
        device : str, optional
            The device where the data lays. Default is 'cpu'.

        Notes
        -----
        To overwrite the data ``current_out`` (the fixed function) just call
        ``.fix_input`` again with a new function.
        """
        if isinstance(function, FunctionSet):
            function.create_functions(device=device)
            index = torch.arange(function.function_set_size)
            fns = function.get_function(index)
            input_points = self.grid.to(device)
            discrete_fn = fns(input_points)
        elif callable(function):
            function = UserFunction(function)
            discrete_points = self.grid.to(device)
            discrete_fn = function(discrete_points)
            discrete_fn = discrete_fn.unsqueeze(0)  # add batch dimension
            discrete_fn = Points(discrete_fn, self.input_space.output_space)
        elif isinstance(function, Points):
            # check if we have to add batch dimension
            if len(function._t.shape) < 3:
                discrete_fn = Points(
                    function._t.unsqueeze(0), self.input_space.output_space
                )
            else:
                discrete_fn = function
        elif isinstance(function, torch.Tensor):
            # check if we have to add batch dimension
            if len(function.shape) < 3:
                discrete_fn = function.unsqueeze(0)
                discrete_fn = Points(discrete_fn, self.input_space.output_space)
            else:
                discrete_fn = Points(function, self.input_space.output_space)
        else:
            raise NotImplementedError(
                "Function has to be callable, a FunctionSet, a tensor, or a tp.Point"
            )
        self(discrete_fn)


class FCBranchNet(BranchNet):
    """A fully connected neural network as a branch net inside a DeepONet-model.

    Parameters
    ----------
    function_space : Space
        The space of functions that can be put in this network.
    grid : torchphysics.spaces.Points
        The points at which the input functions should
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of grid points
        times the function space dimension.
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
        function_space,
        grid,
        hidden=(20, 20, 20),
        activations=nn.Tanh(),
        xavier_gains=5 / 3,
    ):
        super().__init__(function_space, grid)
        self.hidden = hidden
        self.activations = activations
        self.xavier_gains = xavier_gains
        
        self.input_neurons = len(self.grid) * self.input_space.output_space.dim

    def finalize(self, output_space, output_neurons):
        super().finalize(output_space, output_neurons)
        layers = _construct_FC_layers(
            hidden=self.hidden,
            input_dim=self.input_neurons,
            output_dim=self.output_neurons,
            activations=self.activations,
            xavier_gains=self.xavier_gains,
        )

        self.sequential = nn.Sequential(*layers)

    def forward(self, discrete_function_batch):
        discrete_function_batch = discrete_function_batch.as_tensor.reshape(
            -1, self.input_dim
        )
        self.current_out = self._reshape_multidimensional_output(
            self.sequential(discrete_function_batch)
        )


class ConvBranchNet(BranchNet):
    """A branch network that first applies a convolution to the input functions
    and afterwards linear FC-layers.

    Parameters
    ----------
    function_space : Space
        The space of functions that can be put in this network.
    grid : torchphysics.spaces.Points
        The points at which the input functions should be
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of grid points.
    convolutional_network : torch.nn.module
        The user defined convolutional network, that should be applied to the
        branch input. Inside this network, the input can be transformed arbitrary,
        e.g. you can also apply pooling or other layers.
        We only expect that the network gets the input in the shape:

        [batch_dim, function_space.output_space.dim (channels_in),
         len(grid)]

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
        function_space,
        grid,
        convolutional_network,
        hidden=(20, 20, 20),
        activations=nn.Tanh(),
        xavier_gains=5 / 3,
    ):
        super().__init__(function_space, grid)
        self.conv_net = convolutional_network
        self.hidden = hidden
        self.activations = activations
        self.xavier_gains = xavier_gains

    def finalize(self, output_space, output_neurons):
        super().finalize(output_space, output_neurons)
        layers = _construct_FC_layers(
            hidden=self.hidden,
            input_dim=None,
            output_dim=self.output_neurons,
            activations=self.activations,
            xavier_gains=self.xavier_gains,
        )

        self.sequential = nn.Sequential(*layers)

    def forward(self, discrete_function_batch):
        # for convolution we have to change the dimension order of
        # the input.
        # Pytorch conv need: (batch, channels_in, length_1, length_2, ...)
        # Generally we have : (batch, length_1, ..., channels_in), where channels_in
        # corresponds to the output dimension of our functions and length to the
        # number of discretization points. -> move dim. -1 to 1
        discrete_function_batch = discrete_function_batch.as_tensor
        x = self.conv_net(discrete_function_batch.moveaxis(0, -1, 1))
        out = self.sequential(x.flatten(start_dim=1))
        self.current_out = self._reshape_multidimensional_output(out)
