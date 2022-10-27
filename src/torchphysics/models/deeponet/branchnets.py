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
    output_space : Space
        The space of the points that should be
        returned by the parent DeepONet-model.
    output_neurons : int
        The number of output neurons. These neurons will only
        be used internally. Will be multiplied my the dimension of the output space, 
        so each dimension will have the same number of intermediate neurons. 
        The final output of the DeepONet-model will be in the dimension of the 
        output space. 
    discretization_sampler : torchphysics.sampler
        A sampler that will create the points at which the input functions should 
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of sampled points.
        Therefore, the sampler should always return the same number of points!
    """
    def __init__(self, function_space, output_space, output_neurons, 
                 discretization_sampler):
        super().__init__(function_space, output_space)
        self.output_neurons = output_neurons * output_space.dim
        self.discretization_sampler = discretization_sampler
        self.input_dim = len(self.discretization_sampler)
        self.current_out = torch.empty(0)

    def _reshape_multidimensional_output(self, output):
        return output.reshape(-1, self.output_space.dim, 
                              int(self.output_neurons/self.output_space.dim))
        
    @abc.abstractmethod
    def forward(self, discrete_function_batch, device='cpu'):
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

    def _discretize_function_set(self, function_set, device='cpu'):
        """Internal discretization of the training set.
        """
        input_points = self.discretization_sampler.sample_points(device=device)
        #self.input_points = input_points
        fn_out = function_set.create_function_batch(input_points)
        return fn_out

    def fix_input(self, function, device='cpu'):
        """Fixes the branch net for a given function. The branch net will 
        be evaluated for the given function and the output saved in ``current_out``. 
        
        Parameters
        ----------
        function : callable, torchphysics.domains.FunctionSet
            The function(s) for which the network should be evaluaded.
        device : str, optional
            The device where the data lays. Default is 'cpu'.
        
        Notes
        -----
        To overwrite the data ``current_out`` (the fixed function) just call 
        ``.fix_input`` again with a new function.
        """
        # TODO: add  functionality for list of functions and already 
        # discrete function tensor
        if isinstance(function, FunctionSet):
            function.sample_params(device=device)
            discrete_fn = self._discretize_function_set(function, device=device)
        elif callable(function):
            function = UserFunction(function)
            discrete_points = self.discretization_sampler.sample_points(device=device)
            discrete_fn = function(discrete_points)
            discrete_fn = discrete_fn.unsqueeze(0) # add batch dimension
            discrete_fn = Points(discrete_fn, self.input_space.output_space)
        else:
            raise NotImplementedError("function has to be callable or a FunctionSet")
        self(discrete_fn)


class FCBranchNet(BranchNet):
    """A neural network that can be used inside a DeepONet-model.
    
    Parameters
    ----------
    function_space : Space
        The space of functions that can be put in this network.
    output_space : Space
        The space of the points that should be
        returned by the parent DeepONet-model.
    output_neurons : int
        The number of output neurons. These neurons will only
        be used internally. The final output of the DeepONet-model will be 
        in the dimension of the output space. 
    discretization_sampler : torchphysics.sampler
        A sampler that will create the points at which the input functions should 
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of sampled points.
        Therefore, the sampler should always return the same number of points!
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
    def __init__(self, function_space, output_space, output_neurons,
                 discretization_sampler, hidden=(20,20,20), activations=nn.Tanh(),
                 xavier_gains=5/3):
        super().__init__(function_space, output_space, 
                         output_neurons, discretization_sampler)
        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_dim, 
                                      output_dim=self.output_neurons, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, discrete_function_batch):
        discrete_function_batch = discrete_function_batch.as_tensor.reshape(-1, self.input_dim)
        self.current_out = self._reshape_multidimensional_output(self.sequential(discrete_function_batch))
       

class ConvBranchNet1D(BranchNet):
    """A branch network that first applies a convolution to the input functions
    and afterwards linear FC-layers.

    Parameters
    ----------
    function_space : Space
        The space of functions that can be put in this network.
    output_space : Space
        The space of the points that should be
        returned by the parent DeepONet-model.
    output_neurons : int
        The number of output neurons. These neurons will only
        be used internally. The final output of the DeepONet-model will be 
        in the dimension of the output space. 
    discretization_sampler : torchphysics.sampler
        A sampler that will create the points at which the input functions should 
        evaluated, to create a discrete input for the network.
        The number of input neurons will be equal to the number of sampled points.
        Therefore, the sampler should always return the same number of points!
    convolutional_network : torch.nn.module
        The user defined convolutional network, that should be applied to the 
        branch input. Inside this network, the input can be transformed arbitrary, 
        e.g. you can also apply pooling or other layers. 
        We only expect that the network gets the input in the shape:

        [batch_dim, function_space.output_space.dim (channels_in), 
         len(discretization_sampler)]

        You have to make sure, that the number of output dimension is 
        compatible with the following linear layers.
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
    def __init__(self, function_space, output_space, output_neurons,
                 discretization_sampler, convolutional_network,
                 hidden=(20,20,20), activations=nn.Tanh(), xavier_gains=5/3):
        super().__init__(function_space, output_space, 
                         output_neurons, discretization_sampler)
        self.conv_net = convolutional_network
        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_dim, 
                                      output_dim=self.output_neurons, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def _discretize_fn(self, function, device):
        # where is this function used?
        function = UserFunction(function)
        discrete_points = self.discretization_sampler.sample_points(device=device)
        discrete_fn = function(discrete_points)
        if discrete_fn.shape[0] == self.input_dim:
            discrete_fn = discrete_fn.T
        return discrete_fn.unsqueeze(-1)

    def forward(self, discrete_function_batch):
        # for convolution we have to change the dimension order of 
        # the input.
        # Pytorch conv1D needs: (batch, channels_in, length)
        # Generally we have : (batch, length, channels_in), where channels_in
        # corresponds to the output dimension of our functions and length to the 
        # number of discretization points. -> switch dim. 1 and 2
        discrete_function_batch = discrete_function_batch.as_tensor
        x = self.conv_net(discrete_function_batch.permute(0, 2, 1))
        # for the linear layer transform again and remove the last dimension:
        out = self.sequential(x.permute(0, 2, 1).reshape(-1, self.input_dim))
        self.current_out = self._reshape_multidimensional_output(out)