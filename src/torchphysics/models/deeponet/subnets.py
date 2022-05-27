import torch
import torch.nn as nn

from ..model import Model
from ..fcn import _construct_FC_layers
from ...problem.domains.functionsets.functionset import FunctionSet
from ...utils.user_fun import UserFunction


class TrunkNet(Model):

    def __init__(self, input_space, output_space, output_neurons):
        super().__init__(input_space, output_space)
        self.output_neurons = output_neurons


class BranchNet(Model):

    def __init__(self, function_space, output_space, output_neurons, 
                 discretization_sampler):
        super().__init__(function_space, output_space)
        self.output_neurons = output_neurons
        self.discretization_sampler = discretization_sampler
        self.input_dim = len(self.discretization_sampler)
        self.current_out = torch.empty(0)

    def forward(self, discrete_function_batch, device='cpu'):
        """Forward already expects a discrete batch
        """
        pass

    def _discretize_function_set(self, function_set, device):
        """Internal discretization of the trainings set.
        """
        input_points = self.discretization_sampler.sample_points(device=device)
        self.input_points = input_points
        fn_out = function_set.create_function_batch(self.input_points)
        # fn_out will be of the length (len(function_set)*len(discrete_points)).
        # We need the output batchwise like (len(function_set), len(discrete_points)), 
        # to be able to plug it into the network
        return fn_out.as_tensor.reshape(-1, self.input_dim)

    def fix_input(self, function, device='cpu'):
        """Can the user call to fix the branch net for a given function/function_set.
        - Name of methode is maybe not the best
        - Later maybe add functionality for list of functions
        """
        if isinstance(function, FunctionSet):
            function.sample_params(device=device)
            discrete_fn = self._discretize_function_set(function, device=device)
        elif callable(function):
            function = UserFunction(function)
            discrete_points = self.discretization_sampler.sample_points(device=device)
            discrete_fn = function(discrete_points)
            if discrete_fn.shape[0] == self.input_dim:
                discrete_fn = discrete_fn.T
        self(discrete_fn)


class FCTrunkNet(TrunkNet):

    def __init__(self, input_space, output_space, output_neurons, 
                 hidden=(20,20,20), activations=nn.Tanh(), xavier_gains=5/3):
        super().__init__(input_space, output_space, output_neurons)

        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_space.dim, 
                                      output_dim=output_neurons, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, points):
        return self.sequential(points)


class FCBranchNet(BranchNet):

    def __init__(self, function_space, output_space, output_neurons,
                 discretization_sampler, hidden=(20,20,20), activations=nn.Tanh(),
                 xavier_gains=5/3):
        super().__init__(function_space, output_space, 
                         output_neurons, discretization_sampler)
        layers = _construct_FC_layers(hidden=hidden, input_dim=self.input_dim, 
                                      output_dim=output_neurons, 
                                      activations=activations, xavier_gains=xavier_gains)

        self.sequential = nn.Sequential(*layers)

    def forward(self, discrete_function_batch):
        self.current_out = self.sequential(discrete_function_batch)