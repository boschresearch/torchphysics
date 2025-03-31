import abc
import torch


integer_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int, torch.int64, torch.long]

class FunctionSet():
    """ A function set describes a specfic type of functions that can be used 
    for creating data for training different operator approaches.

    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace
        The function space that the functions in the set should be part of.
        This defines what input and output space the functions have.
    function_set_size : int
        An integer that defines how many functions are stored in the set. 
        This is used for creating multiple functions at once and
        then storing them for later use.

    Notes
    -----
    The `function_set_size` is motivated by the case that creating functions 
    (either by computations or loading from disk) can be computationally 
    expensive if done for each function individually. Therefore, we aim
    to create multiple functions at once and then store them. When
    later functions are sampled from this set, they can be quickly returned.
    But this creation is not only done once but can be repeated after
    some time to obtain new functions.
    """
    def __init__(self, function_space, function_set_size):
        self.function_space = function_space
        self.function_set_size = function_set_size
    
    @property
    def is_discretized(self):
        """ Returns if the function set is already discretized.
        """
        return False
    
    def is_discretization_of(self, function_set):
        """ Returns if the function set is the discretization of another 
        function set.

        Parameters
        ----------
        function_set : tp.domains.FunctionSet
            The other function set we should compare with.
        """
        return False
    
    @abc.abstractmethod
    def create_functions(self, device="cpu"):
        """ Creates the functions for the function set and stores them.
        The created functions can then be retrieved by the `get_function` method.

        Parameters
        ----------
        device : str
            The device on which the functions should be stored.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_function(self, idx):
        """ Returns a function from the function set.

        Parameters
        ----------
        idx : int or list, tensor of int
            The index of the function that should be returned.
            Multiple functions can be returned at once when a list or tensor
            is passed in. Note that the index should be in the range of the function
            set size.

        Returns
        -------
        callable or torch.tensor
            Returns the function. If the function set is discrete, the functions
            can not be further evaluated and are therefore returned as a tensor. Otherwise
            a callable is returned that can be evaluated at any point.
        """
        raise NotImplementedError
    
    def discretize(self, locations):
        """ Discretizes the function set, to be always evaluated at the 
        provided locations.

        Parameters
        ----------
        locations : tp.spaces.Points
            The points at which the functions should be evaluated.

        Returns
        -------
        tp.domains.DiscretizedFunctionSet
            The discretized function set.
        """
        assert len(locations.as_tensor.shape) >= 3, \
            f"""Locations for discretization need a shape >= 3 to be compatible with all implemented
                methods. The provided shape is only of length {len(locations.as_tensor.shape)}. 
                Apply .unsqueeze(0) to the data to add one additional batch dimension."""
        return DiscretizedFunctionSet(self, locations)

    def __mul__(self, other):
        """ Creates a product of two function sets. Leading to a product in the
        function spaces and the function outputs are concatenated along the output
        dimension.

        Parameters
        ----------
            The other function set that should be multiplied with this one.

        Returns
        -------
        tp.domains.FunctionSetProduct
            The product of the two function sets.
        """
        from .functionset_operations import FunctionSetProduct

        assert self.function_space.output_space != other.function_space.output_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        
        if isinstance(other, FunctionSetProduct):
            return other * self
        if other.is_discretized and not self.is_discretized:
            return other * self
        else:
            assert self.function_set_size == other.function_set_size, \
                """Both FunctionSets need the same size!"""
            return FunctionSetProduct(self.function_space*other.function_space, [self, other])

    def append(self, other):
        """ Stacks two function sets together, such that different kind of functions
        can be combined into one set.

        Parameters
        ----------
            The other function set that should be connected with this one.

        Returns
        -------
        tp.domains.FunctionSetCollection
            The collection of the two function sets.
        """
        from .functionset_operations import FunctionSetCollection

        assert self.function_space.output_space == other.function_space.output_space, \
                """Both FunctionSets need the same output space!"""
        if isinstance(other, FunctionSetCollection):
            return other * self
        else:
            return FunctionSetCollection(self.function_space, [self, other])

    def __add__(self, other):
        """ Performs the "pointwise" addition of two function sets.

        Parameters
        ----------
            The other function set that should be added to this one.

        Returns
        -------
        tp.domains.FunctionSetAdd
            The function sets that computes the sum of the inputs.
        """
        from .functionset_operations import FunctionSetAdd

        if isinstance(other, FunctionSetAdd):
            return other + self
        else:
            return FunctionSetAdd(self.function_space, [self, other])

    def __sub__(self, other):
        """ Performs the "pointwise" substraction of two function sets.

        Parameters
        ----------
            The other function set that should be substracted from this one.

        Returns
        -------
        tp.domains.FunctionSetSubstract
            The function sets that computes the difference of the inputs.
        """
        from .functionset_operations import FunctionSetSubstract
        return FunctionSetSubstract(self.function_space, [self, other])


class DiscretizedFunctionSet(FunctionSet):
    """ A discretized function set that is always evaluated at the provided locations.

    Parameters
    ----------
    function_set : tp.domains.FunctionSet
        The function set that should be discretized.
    locations : tp.spaces.Points
        The points at which the functions should be evaluated.

    Note
    ----
    This class is not fully functional yet!
    """
    def __init__(self, function_set : FunctionSet, locations):
        super().__init__(function_set.function_space, function_set.function_set_size)
        self.function_set = function_set
        self.locations = locations
        if self.function_set.is_discretized:
            assert self.locations.dtype in integer_dtypes

    @property
    def is_discretized(self):
        return True
    
    def is_discretization_of(self, function_set):
        return (self.function_set is function_set) or (self.function_set.is_discretization_of(function_set))
    
    def create_functions(self, device="cpu"):
        self.locations = self.locations.to(device)
        self.function_set.create_functions(device)

    def get_function(self, idx):
        samples = self.function_set.get_function(idx)
        if callable(samples):
            return samples(self.locations)
        else:
            # we assume that self.locations is a grid, and its last dimension corresponds
            # to the amount of grid axis. i.e.
            assert (len(samples.shape) - 2) == self.locations.shape[-1]
            out_shape = (samples.shape[0], *self.locations.shape[0:-1], samples.shape[-1])
            locations_slice = torch.unbind(torch.reshape(self.locations,
                                                         (-1, self.locations.shape[-1])),
                                           dim=-1)
            locations_slice = (slice(None), *locations_slice, slice(None))
            return samples[locations_slice].reshape(*out_shape)
    
    def discretize(self, locations):
        assert torch.is_tensor(locations)
        assert locations.dtype in integer_dtypes, \
            """A discretized FunctionSet can only be further discretized by passing in indices 
                to subsample the current discretization."""
        return DiscretizedFunctionSet(self, locations)


    def __mul__(self, other):
        from .functionset_operations import FunctionSetProduct
        
        assert self.function_space.output_space != other.function_space.output_space, \
                """Both FunctionSets have the same output space, maybe you want to use 'append' instead?"""
        assert self.function_set_size == other.function_set_size, \
                """Both FunctionSets need the same size!"""
        
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
