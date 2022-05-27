import abc
import torch

from ...spaces.points import Points
from ....utils.user_fun import UserFunction
from ...spaces.functionspace import FunctionSpace


class FunctionSet():
    """
    A set of functions that can be sampled to supply samples from a function space.
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
        """
        assert other.function_space == self.function_space, \
            """Both FunctionSets do not have the same FunctionSpace!"""
        if isinstance(other, FunctionSetCollection):
            return other + self
        else:
            return FunctionSetCollection([self, other])

    def sample_params(self, device='cpu'):
        """Samples parameters of the function space.
        """
        self.param_batch = self.parameter_sampler.sample_points(device=device)
    
    def create_function_batch(self, points):
        """Evaluates the underlying function object to create a batch of 
        discrete function samples.
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
        #param_point_meshgrid = Points(torch.repeat_interleave(self.param_batch,
        #                                                      n_points, dim=0), 
        #                              self.param_batch.space)
        points_repeated = points.as_tensor.unsqueeze(0).repeat(n_params,1,1)
        params_repeated = self.param_batch.as_tensor.unsqueeze(1).repeat(1,n_points,1)
        param_point_meshgrid = Points(torch.cat((params_repeated, points_repeated), dim=-1),
                                      self.param_batch.space*points.space)
        return param_point_meshgrid#.join(points.repeat(n_params))

    @abc.abstractmethod
    def _evaluate_function(self, param_point_meshgrid):
        """Here the underlying functions of the FunctionDomain will be evaluated. 
        """
        raise NotImplementedError


class FunctionSetCollection(FunctionSet):
    """Collection of multiple FunctionSets (for __add__)
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

    def sample_params(self, device='cpu'):
        for function_set in self.collection:
            function_set.sample_params(device)
    
    def create_function_batch(self, points):
        output = Points.empty()
        for function_set in self.collection:
            output = output | function_set.create_function_batch(points)
        return output


class CustomFunctionSet(FunctionSet):
    """FunctionSet for a arbitrary "basis function"
    """
    def __init__(self, function_space, parameter_sampler, basis_fn):
        super().__init__(function_space=function_space, 
                         parameter_sampler=parameter_sampler)
        if not isinstance(basis_fn, UserFunction):
            basis_fn = UserFunction(basis_fn)
        self.basis_fn = basis_fn

    def _evaluate_function(self, param_point_meshgrid):
        return self.basis_fn(param_point_meshgrid)