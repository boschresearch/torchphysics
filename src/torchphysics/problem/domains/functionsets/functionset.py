import abc
import torch

from ...spaces.points import Points
from ....utils.user_fun import UserFunction
from ...spaces.functionspace import FunctionSpace


class FunctionSet():
    """A set of functions that can supply samples from a function space.

    Parameters
    ----------
    function_space : torchphysics.spaces.FunctionSpace
        The space of which this set of functions belongs to. The inputs and outputs
        of this FunctionSet are defined by the corresponding values inside the function
        space.
    parameter_sampler : torchphysics.samplers.PointSampler
        A sampler that provides additional parameters that can be used 
        to create different kinds of functions. E.g. our FunctionSet consists
        of Functions like k*x, x is the input variable and k is given through 
        the sampler. 

        During each training iteration will call the parameter_sampler to sample
        new parameters. For each parameter a function will be created and the 
        input batch of functions will be of the same length as the sampled 
        parameters.
    """
    def __init__(self, function_space, parameter_sampler):
        assert isinstance(function_space, FunctionSpace), \
            """A FunctionSet needs a torchphysics.spaces.FunctionSpace!"""
        self.function_space = function_space
        self.parameter_sampler = parameter_sampler
        self.param_batch = None
        self.current_iteration_num = -1

    def __add__(self, other):
        """Combines two function sets. 
        
        Notes
        -----
        When parameters are sampled, will sample them from both sets. 
        Creates a batch of functions consisting of the batch of each set.
        (Length of the batches will be added)
        """
        assert other.function_space == self.function_space, \
            """Both FunctionSets do not have the same FunctionSpace!"""
        if isinstance(other, FunctionSetCollection):
            return other + self
        else:
            return FunctionSetCollection([self, other])
    
    def __len__(self):
        """Returns the amount of functions sampled in a single call to sample_params.
        """
        return len(self.parameter_sampler)

    def sample_params(self, device='cpu'):
        """Samples parameters of the function space.

        Parameters
        ----------
        device : str, optional
            The device, where the parameters should be created. Default is 'cpu'.

        Notes
        -----
        We save the sampled parameters internally, so that we can use them multiple times.
        Since given a parameter we still have a continuous representation of the underlying
        function types. When the functions should be evaluated at some input points, 
        we just have to create the meshgrid of parameters and points.
        """
        self.param_batch = self.parameter_sampler.sample_points(device=device)
    
    def create_function_batch(self, points):
        """Evaluates the underlying function object to create a batch of 
        discrete function samples.

        Parameters
        ----------
        points : torchphysics.spaces.Points
            The input points, where we want to evaluate a set of functions.

        Returns
        -------
        torchphysics.spaces.Points
            The batch of discrete function samples. The underlying tensor is of the
            shape: [len(self), len(points), self.function_space.output_space.dim]
        """
        param_point_meshgrid = self._create_meshgrid(points)
        output = self._evaluate_function(param_point_meshgrid)
        return Points(output, self.function_space.output_space)

    def _create_meshgrid(self, points):
        """Creates the meshgrid of current batch parameter and discretization
        points.
        """
        n_points = len(points)
        n_params = len(self.param_batch)
        points_repeated = points.as_tensor.unsqueeze(0).repeat(n_params,1,1)
        params_repeated = self.param_batch.as_tensor.unsqueeze(1).repeat(1,n_points,1)
        param_point_meshgrid = Points(torch.cat((params_repeated, points_repeated), dim=-1),
                                      self.param_batch.space*points.space)
        return param_point_meshgrid

    @abc.abstractmethod
    def _evaluate_function(self, param_point_meshgrid):
        """Here the underlying functions of the FunctionSet will be evaluated. 
        """
        raise NotImplementedError


class FunctionSetCollection(FunctionSet):
    """Collection of multiple FunctionSets. Used for the additions of 
    different FunctionSets.

    Parameters
    ----------
    function_sets : list, tuple
        A list/tuple of FunctionSets.
    """
    def __init__(self, function_sets):
        self.collection = function_sets
        super().__init__(function_space=function_sets[0].function_space,
                         parameter_sampler=None)

    def __add__(self, other):
        assert other.function_space == self.function_space, \
            """Both FunctionSets do not have the same FunctionSpace!"""
        if isinstance(other, FunctionSetCollection):
            self.collection += other.collection
        else:
            self.collection.append(other)
        return self
    
    def __len__(self):
        return sum(len(f_s) for f_s in self.collection)

    def sample_params(self, device='cpu'):
        for function_set in self.collection:
            function_set.sample_params(device)
    
    def create_function_batch(self, points):
        output = Points.empty()
        for function_set in self.collection:
            output = output | function_set.create_function_batch(points)
        return output


class CustomFunctionSet(FunctionSet):
    """FunctionSet for an arbitrary function.

    Parameters
    ----------
    function_space : torchphysics.spaces.FunctionSpace
        The space of which this set of functions belongs to. The inputs and outputs
        of this FunctionSet are defined by the corresponding values inside the function
        space.
    parameter_sampler : torchphysics.samplers.PointSampler
        A sampler that provides additional parameters that can be used 
        to create different kinds of functions. E.g. our FunctionSet consists
        of Functions like k*x, x is the input variable and k is given through 
        the sampler. 

        During each training iteration will call the parameter_sampler to sample
        new parameters. For each parameter a function will be created and the 
        input batch of functions will be of the same length as the sampled 
        parameters.
    custom_fn : callable
        A function that describes the FunctionSet. The input of the functions 
        can include the variables of the function_space.input_space and the 
        parameters from the parameter_sampler.
    """
    def __init__(self, function_space, parameter_sampler, custom_fn):
        super().__init__(function_space=function_space, 
                         parameter_sampler=parameter_sampler)
        if not isinstance(custom_fn, UserFunction):
            custom_fn = UserFunction(custom_fn)
        self.custom_fn = custom_fn

    def _evaluate_function(self, param_point_meshgrid):
        return self.custom_fn(param_point_meshgrid)