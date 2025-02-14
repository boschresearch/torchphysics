import abc
import torch

from ...spaces.points import Points
from ...spaces.functionspace import FunctionSpace


class FunctionSet:
    """A set of functions that can supply samples from a function space.
    
    Parameters
    ----------
    function_space : torchphysics.spaces.FunctionSpace
        The space of which this set of functions belongs to. The inputs and outputs
        of this FunctionSet are defined by the corresponding values inside the function
        space.
    """
    def __init__(self, function_space):
        assert isinstance(
            function_space, FunctionSpace
        ), """A FunctionSet needs a torchphysics.spaces.FunctionSpace!"""
        self.function_space = function_space

    @abc.abstractmethod
    def sample_functions(self, n_samples, locations, device="cpu"):
        """Samples functions from the function space.

        Parameters
        ----------
        n_samples : int
            The amount of functions to sample.
        locations : torchphysics.spaces.Points, list, tuple
            The locations where the functions should be evaluated. Each function from
            the batch will be evaluated at these locations.
            If it is a list/tuple of Points, the functions will be evaluated at each
            element of the list/tuple.
            The points should contain data of either 
            [1, number_of_locations, input_dim] or [n_samples, number_of_locations, input_dim].
        device : str, optional
            The device, where the functions should be created. Default is 'cpu'.

        Returns
        -------
        torchphysics.spaces.Points
            ....
        """
        pass

    def __mul__(self, other):
        """Creates the union of two function sets, by concatenating the output of
        both FunctionSets. Therefore, we will sample functions in the product space.

        """
        assert self.function_space.output_space != other.function_space.outer_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        
        if isinstance(other, FunctionSetUnion):
            return other * self
        else:
            return FunctionSetUnion(self.function_space*other.function_space, [self, other])


    def append(self, other):
        """Appends the output of two function sets, to one larger batch of functions.
        """
        assert self.function_space.output_space == other.function_space.outer_space, \
                """Both FunctionSets need the same output space!"""
        
        if isinstance(other, FunctionSetCollection):
            return other.append(self)
        else:
            return FunctionSetCollection(self.function_space*other.function_space, [self, other])
        

class FunctionSetUnion(FunctionSet):
    """Collection of multiple FunctionSets. Used for creating functions in the product 
    space.
    """
    def __init__(self, function_space, function_sets):
        super().__init__(function_space)
        self.function_sets = function_sets


    def __mul__(self, other):
        assert self.function_space.output_space != other.function_space.outer_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        if isinstance(other, FunctionSetUnion):
            return FunctionSetUnion(self.function_space*other.function_space, 
                                    self.function_sets + other.function_sets)
        else:
            return FunctionSetUnion(self.function_space*other.function_space, 
                                    self.function_sets + [other])

    def sample_functions(self, n_samples, locations, device="cpu"):
        outputs = []
        for function_set in self.function_sets:
            outputs.append(function_set.sample_functions(n_samples, locations, device))
        # For one batch of inputs we return just a single output
        if isinstance(locations, Points):
            return Points.joined(*outputs)
        # If we have multiple batches of inputs we return the functions evaluated at each input.
        # In this case "outputs" is a list of tuples, where each tuple contains the outputs of each
        # function set.
        output_per_locations = list(zip(*outputs))
        final_output = []
        for i in output_per_locations:
            final_output.append(Points.joined(*i))
        return final_output


class FunctionSetCollection(FunctionSet):
    """Collection of multiple FunctionSets. Used for creating a batch of functions.
    """
    def __init__(self, function_space, function_sets):
        super().__init__(function_space)
        self.function_sets = function_sets

    def append(self, other):
        assert self.function_space.output_space == other.function_space.outer_space, \
                """Both FunctionSets need the same output space!"""
        
        if isinstance(other, FunctionSetCollection):
            return FunctionSetCollection(self.function_space*other.function_space,
                                         self.function_sets + other.function_sets)
        else:
            return FunctionSetCollection(self.function_space*other.function_space, 
                                         self.function_sets + [other])
        
    def sample_functions(self, n_samples, locations, device="cpu"):
        n_samples_per_set = n_samples // len(self.function_sets)
        outputs = []
        # Splitting as in the case of FunctionSetUnion:
        for idx, function_set in enumerate(self.function_sets):
            # Last set might have less samples so we have n_samples in total
            if idx == len(self.function_sets) - 1:
                outputs.append(
                    function_set.sample_functions(n_samples - idx*n_samples_per_set, 
                                                locations, device))
            else:
                outputs.append(
                    function_set.sample_functions(n_samples_per_set, 
                                                locations, device))
        # Different cases as for the FunctionSetUnion:
        if isinstance(locations, Points):
            return Points(torch.cat([out.as_tensor for out in outputs], dim=0), 
                          self.function_space.output_space)
        else:
            output_per_locations = list(zip(*outputs))
            final_output = []
            for i in output_per_locations:
                final_output.append(
                    Points(torch.cat([out.as_tensor for out in outputs], dim=0), 
                           self.function_space.output_space)
                )
            return final_output

# class FunctionSetOld:
#     """A set of functions that can supply samples from a function space.

#     Parameters
#     ----------
#     function_space : torchphysics.spaces.FunctionSpace
#         The space of which this set of functions belongs to. The inputs and outputs
#         of this FunctionSet are defined by the corresponding values inside the function
#         space.
#     parameter_sampler : torchphysics.samplers.PointSampler
#         A sampler that provides additional parameters that can be used
#         to create different kinds of functions. E.g. our FunctionSet consists
#         of Functions like k*x, x is the input variable and k is given through
#         the sampler.

#         During each training iteration will call the parameter_sampler to sample
#         new parameters. For each parameter a function will be created and the
#         input batch of functions will be of the same length as the sampled
#         parameters.
#     """

#     def __init__(self, function_space, parameter_sampler):
#         assert isinstance(
#             function_space, FunctionSpace
#         ), """A FunctionSet needs a torchphysics.spaces.FunctionSpace!"""
#         self.function_space = function_space
#         self.parameter_sampler = parameter_sampler
#         self.param_batch = None
#         self.current_iteration_num = -1

#     def __add__(self, other):
#         """Combines two function sets.

#         Notes
#         -----
#         When parameters are sampled, will sample them from both sets.
#         Creates a batch of functions consisting of the batch of each set.
#         (Length of the batches will be added)
#         """
#         assert (
#             other.function_space == self.function_space
#         ), """Both FunctionSets do not have the same FunctionSpace!"""
#         if isinstance(other, FunctionSetCollection):
#             return other + self
#         else:
#             return FunctionSetCollection([self, other])

#     def __len__(self):
#         """Returns the amount of functions sampled in a single call to sample_params."""
#         return len(self.parameter_sampler)

#     def sample_params(self, device="cpu"):
#         """Samples parameters of the function space.

#         Parameters
#         ----------
#         device : str, optional
#             The device, where the parameters should be created. Default is 'cpu'.

#         Notes
#         -----
#         We save the sampled parameters internally, so that we can use them multiple times.
#         Since given a parameter we still have a continuous representation of the underlying
#         function types. When the functions should be evaluated at some input points,
#         we just have to create the meshgrid of parameters and points.
#         """
#         self.param_batch = self.parameter_sampler.sample_points(device=device)

#     def create_function_batch(self, points):
#         """Evaluates the underlying function object to create a batch of
#         discrete function samples.

#         Parameters
#         ----------
#         points : torchphysics.spaces.Points
#             The input points, where we want to evaluate a set of functions.

#         Returns
#         -------
#         torchphysics.spaces.Points
#             The batch of discrete function samples. The underlying tensor is of the
#             shape: [len(self), len(points), self.function_space.output_space.dim]
#         """
#         param_point_meshgrid = self._create_meshgrid(points)
#         output = self._evaluate_function(param_point_meshgrid)
#         return Points(output, self.function_space.output_space)

#     def _create_meshgrid(self, points):
#         """Creates the meshgrid of current batch parameter and discretization
#         points.
#         """
#         n_points = len(points)
#         n_params = len(self.param_batch)
#         points_repeated = points.as_tensor.unsqueeze(0).repeat(n_params, 1, 1)
#         params_repeated = self.param_batch.as_tensor.unsqueeze(1).repeat(1, n_points, 1)
#         param_point_meshgrid = Points(
#             torch.cat((params_repeated, points_repeated), dim=-1),
#             self.param_batch.space * points.space,
#         )
#         return param_point_meshgrid

#     @abc.abstractmethod
#     def _evaluate_function(self, param_point_meshgrid):
#         """Here the underlying functions of the FunctionSet will be evaluated."""
#         raise NotImplementedError


# class FunctionSetCollection(FunctionSet):
#     """Collection of multiple FunctionSets. Used for the additions of
#     different FunctionSets.

#     Parameters
#     ----------
#     function_sets : list, tuple
#         A list/tuple of FunctionSets.
#     """

#     def __init__(self, function_sets):
#         self.collection = function_sets
#         super().__init__(
#             function_space=function_sets[0].function_space, parameter_sampler=None
#         )

#     def __add__(self, other):
#         assert (
#             other.function_space == self.function_space
#         ), """Both FunctionSets do not have the same FunctionSpace!"""
#         if isinstance(other, FunctionSetCollection):
#             self.collection += other.collection
#         else:
#             self.collection.append(other)
#         return self

#     def __len__(self):
#         return sum(len(f_s) for f_s in self.collection)

#     def sample_params(self, device="cpu"):
#         for function_set in self.collection:
#             function_set.sample_params(device)

#     def create_function_batch(self, points):
#         output = Points.empty()
#         for function_set in self.collection:
#             output = output | function_set.create_function_batch(points)
#         return output




# class TestFunctionHelper(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, expected_out, grad_out):
#         ctx.save_for_backward(grad_out)
#         x_ten = torch.sum(x, dim=-1, keepdim=True)
#         return expected_out + 0.0 * x_ten# <- hack to build graph to allow for precomputed gradient

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_out, = ctx.saved_tensors
#         repeats = grad_output.shape[0] // grad_out.shape[0]
#         # Assumes the original data to be repeated along the first axis
#         # TODO: Can be done nicer???
#         return grad_out.repeat((repeats, 1, 1)) * grad_output, None, None


# class TestFunctionSet(FunctionSet):

#     def __init__(self, function_space):
#         super().__init__(function_space=function_space, parameter_sampler=None)
#         self.eval_fn_helper = TestFunctionHelper()
#         self.quadrature_mode_on = True

#     @abc.abstractmethod
#     def switch_quadrature_mode_on(self, set_on : bool):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def __call__(self, x):
#         raise NotImplementedError
    
#     @abc.abstractmethod
#     def to(self, device):
#         raise NotImplementedError
    
#     @abc.abstractmethod
#     def get_quad_weights(self, n):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def get_quadrature_points(self):
#         raise NotImplementedError