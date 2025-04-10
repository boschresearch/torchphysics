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

    def _transform_locations(self, locations):
        # TODO: Improve this for general location shapes
        if len(locations.shape) == 1:
            locations = locations.unsqueeze(0)
            
        if locations.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(
                locations[self.function_space.input_space].as_tensor, 
                len(self.current_idx), dim=0
                )
        else:
            location_copy = locations[self.function_space.input_space].as_tensor[:len(self.current_idx)]
        return location_copy


class DiscreteFunctionSet(FunctionSet):
    """ A function set that only returns already discretized functions, which
    can not be evaluated at arbitrary locations.
    """
    def __init__(self, function_space, function_set_size, data_shape):
        super().__init__(function_space, function_set_size)
        self.data_shape = data_shape
        self.pca = None
        self.mean_tensor = None
        self.std_tensor = None

    @property
    def is_discretized(self):
        return True

    def discretize(self, locations):
        assert torch.is_tensor(locations)
        assert locations.dtype in integer_dtypes, \
            """A discrete FunctionSet can only be further discretized by passing in indices 
                to subsample the current discretization."""
        return DiscretizedFunctionSet(self, locations)

    def compute_pca(self, 
                    components : int, 
                    normalize_data : bool = True, 
                    ):
        """ Carries out the principal component analysis for this function set.

        Parameters
        ----------
        components : int
            The number of components that should be keeped in the PCA.
        normalize_data : bool, optional
            If the data of the function set should be normalized before the
            PCA is computed (recommented). Default is true.
            Note, the normalization is only applied during this method and 
            not saved afterwards, therefore the underlying data in this function
            set is **not** modified!
            
        Notes
        -----
        The PCA is not returned but instead saved internally for later
        usage. Use '.principal_components' to obtain the PCA.

        Also the data of the function set is flattened over all dimensions expect 
        of the batch dimension. For higher dimensional data (e.g. images) other
        approaches (like localized PCAs on small patches) may be better suited. 
        """
        pass

    @property
    def principal_components(self):
        """ Returns the principal components of this function set.
        It is requiered to first call 'compute_pca' to compute them and set
        a number n of the used components.

        Returns
        -------
        list : 
            A list of the principal components in the shape of (U, S, V).
            - U is the matrix of the left singular vectors of shape 
              (function_set_size, n)
            - S is a vector containing the first n eigen values of the 
              covariance matrix.
            - V is the matrix of the principal directions of shape
              (function_set_dimension, n)
            See also: https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html  
        """
        if self.pca is None:
            raise AssertionError("PCA needs to be computed! Use the method 'compute_pca'.")
        return self.pca

    @property
    def mean(self):
        if self.mean_tensor is None: self.compute_normalization()
        return self.mean_tensor

    @property
    def std(self):
        if self.std_tensor is None: self.compute_normalization()
        tol = 0.0001 # (add small value to std in case it is zero,
        # can happen if we have some constant data (Dirichlet condition on boundary))
        return self.std_tensor + tol

    def compute_normalization(self):
        """ Computes the mean and standard deviation over the data contained in this
        function set.
        
        Notes
        -----
        The values are not returned but stored internally.
        Use '.mean' and '.std' to obtain them.
        """
        pass



class DiscretizedFunctionSet(DiscreteFunctionSet):
    """ A discretized function set that is always evaluated at the provided locations.

    Parameters
    ----------
    function_set : tp.domains.FunctionSet
        The function set that should be discretized.
    locations : tp.spaces.Points or torch.tensor
        The points at which the functions should be evaluated.
    """
    def __init__(self, function_set : FunctionSet, locations):
        if torch.is_tensor(locations):
            data_shape = locations.shape[:-1] # dimension of input points not needed
        else:
            data_shape = locations.shape

        super().__init__(function_set.function_space, 
                         function_set.function_set_size, 
                         data_shape)
        
        self.function_set = function_set
        self.locations = locations
        if self.function_set.is_discretized:
            assert self.locations.dtype in integer_dtypes
    
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
