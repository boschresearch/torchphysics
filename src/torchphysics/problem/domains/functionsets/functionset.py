import abc
import torch

from ...spaces.points import Points
from ...spaces.functionspace import FunctionSpace

class FunctionSet():
    def __init__(self, function_space, function_set_size):
        self.function_space = function_space
        self.function_set_size = function_set_size
    
    @property
    def is_discretized(self):
        return False
    
    def is_discretization_of(self, function_set):
        return False
    
    def create_functions(self, device="cpu"):
        # creates a new set of functions of the size "function_set_size"
        # -> we can create more functions at once in the case this is computationally more efficient
        # (for example also usefull when loading functions from disk)
        # The functions are then saved and can be retrieved by the get_function method
        pass

    def get_function(self, idx):
        # returns a lambda function if the function set is continuous
        # returns a tensor if the function set is discrete
        pass
    
    def discretize(self, locations):
        # locations is a tensor of Points or indices
        return DiscretizedFunctionSet(self, locations)

    def __mul__(self, other):
        """Creates the union of two function sets
        """
        assert self.function_space.output_space != other.function_space.output_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        
        if isinstance(other, FunctionSetProduct):
            return other * self
        if other.is_discretized:
            return other * self
        else:
            assert self.function_set_size == other.function_set_size, \
                """Both FunctionSets need the same size!"""
            return FunctionSetProduct(self.function_space*other.function_space, [self, other])


class DiscretizedFunctionSet(FunctionSet):
    def __init__(self, function_set : FunctionSet, locations):
        super().__init__(function_set.function_space, function_set.function_set_size)
        self.function_set = function_set
        self.locations = locations

    @property
    def is_discretized(self):
        return True
    
    def is_discretization_of(self, function_set):
        return (self.function_set is function_set) or (self.function_set.is_discretization_of(function_set))
    
    def create_functions(self, device="cpu"):
        self.function_set.create_functions(device)

    def get_function(self, idx):
        samples = self.function_set.get_function(idx)
        if callable(samples):
            return samples(self.locations)
        else:
            return samples[..., self.locations] # TODO: which dimensions are correct here?
    
    def discretize(self, locations):
        assert torch.is_tensor(locations)
        assert locations.dtype in [torch.int64, torch.long], \
            """A discrtized FunctionSet can only be further discretized by passing in indices to 
                to subsample the current discretization."""
        # We need to now check how the functions should be subsampled, e.g.
        # what kind of indices the used handed in and if the are comptabile with 
        # the kind of functions we are creating/using.
        if self.function_space.input_space.dimension == 1:
            assert len(locations.shape) == 1, \
                """Only needs a 1D tensor of indices for a 1D input space."""
            return DiscretizedFunctionSet(self, locations)
        else:
            # In higher dimension the input space could also not only be flattened 
            # into one tensor dimension but instead be distributed over multiple dimensions
            # (e.g., for 2D a we save the functions as an "image").
            # Therefore, load one example of the data a check the shape.
            self.create_functions()
            test_fn = self.get_function(0) # <- is already a discete tensor
            if len(locations.shape) == 1:
                if len(test_fn.shape) - 2 == 1: # <- remove batch and output space dim.
                    return DiscretizedFunctionSet(self, locations)
                else:
                    Warning("""The shape of the discretized functions is not 1D, but the used indices are.
                            We assume that the indices are meant for discretaizing all dimensions the
                            same way and will build a meshgrid of the indices.
                            """)
                    
                # TODO: Add code here that does this...
            elif len(locations) == len(test_fn.shape) - 2:
                return DiscretizedFunctionSet(self, locations)
            else:
                raise ValueError("""The indices provided do not match the shape of the functions""")


    def __mul__(self, other):
        assert self.function_space.output_space != other.function_space.output_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        
        if isinstance(other, FunctionSetProduct):
            return other * self
        if other.is_discretized:
            assert torch.equal(self.locations, other.locations), \
                """Both DiscretizedFunctionSets need the same locations for creating the product!"""
            return FunctionSetProduct(self.function_space*other.function_space, [self, other])
        else:
            Warning(f"""DiscretizedFunctionSet is multiplied with a continuous FunctionSet.
                    The continuous FunctionSet will be discrtized to create the product.""")
            other_discrete = other.discretize(self.locations)
            return FunctionSetProduct(self.function_space*other.function_space, [self, other_discrete])



class FunctionSetProduct(FunctionSet):
    """Collection of multiple FunctionSets. Used for creating functions in the product 
    space.
    """
    def __init__(self, function_space, function_sets):
        super().__init__(function_space, function_sets[0].function_set_size)
        self.function_sets = function_sets

    @property
    def is_discretized(self):
        return self.function_sets[0].is_discretized

    def __mul__(self, other):
        assert self.function_space.output_space != other.function_space.output_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        # TODO: Check if other space is discrete.....
        if isinstance(other, FunctionSetProduct):
            return FunctionSetProduct(self.function_space*other.function_space, 
                                    self.function_sets + other.function_sets)
        else:
            return FunctionSetProduct(self.function_space*other.function_space, 
                                    self.function_sets + [other])

    def is_discretization_of(self, function_set):
        if isinstance(function_set, FunctionSetProduct):
            for self_set in self.function_sets:
                for other_set in function_set.function_sets:
                    if self_set.is_discretization_of(other_set):
                        break
                else:
                    return False
            return True
        # Can not be a discretization of a non-product function set
        return False
    
    def create_functions(self, device="cpu"):
        for fn_set in self.function_sets:
            fn_set.create_functions(device)

    def get_function(self, idx):
        if self.is_discretized:
            point_list = [fn_set.get_function(idx) for fn_set in self.function_sets]
            return Points.joined(*point_list)
        else:
            self.fn_list = [fn_set.get_function(idx) for fn_set in self.function_sets]
            return self._evaluate_product
    
    def discretize(self, locations):
        assert not self.is_discretized, """FunctionSetProduct is already discretized!"""
        return FunctionSetProduct(self.function_space, 
                                  [f.discretize(locations) for f in self.function_sets])


    def _evaluate_product(self, locations):
        point_list = [fn(locations) for fn in self.fn_list]
        return Points.joined(*point_list)


# class FunctionSetCollection(FunctionSet):
#     """Collection of multiple FunctionSets. Used for creating a batch of functions.
#     """
#     def __init__(self, function_space, function_sets):
#         super().__init__(function_space)
#         self.function_sets = function_sets

#     def append(self, other):
#         assert self.function_space.output_space == other.function_space.output_space, \
#                 """Both FunctionSets need the same output space!"""
        
#         if isinstance(other, FunctionSetCollection):
#             return FunctionSetCollection(self.function_space*other.function_space,
#                                          self.function_sets + other.function_sets)
#         else:
#             return FunctionSetCollection(self.function_space*other.function_space, 
#                                          self.function_sets + [other])
        
#     def sample_functions(self, n_samples, locations, device="cpu"):
#         n_samples_per_set = n_samples // len(self.function_sets)
#         outputs = []
#         # Splitting as in the case of FunctionSetProduct:
#         for idx, function_set in enumerate(self.function_sets):
#             # Last set might have less samples so we have n_samples in total
#             if idx == len(self.function_sets) - 1:
#                 outputs.append(
#                     function_set.sample_functions(n_samples - idx*n_samples_per_set, 
#                                                 locations, device))
#             else:
#                 outputs.append(
#                     function_set.sample_functions(n_samples_per_set, 
#                                                 locations, device))
#         # Different cases as for the FunctionSetProduct:
#         if isinstance(locations, Points):
#             return Points(torch.cat([out.as_tensor for out in outputs], dim=0), 
#                           self.function_space.output_space)
#         else:
#             output_per_locations = list(zip(*outputs))
#             final_output = []
#             for i in output_per_locations:
#                 final_output.append(
#                     Points(torch.cat([out.as_tensor for out in outputs], dim=0), 
#                            self.function_space.output_space)
#                 )
#             return final_output

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