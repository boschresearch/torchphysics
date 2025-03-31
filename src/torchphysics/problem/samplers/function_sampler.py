import abc
import torch

from ..domains.functionsets import FunctionSet


class FunctionSampler:
    """Handles the sampling of functions from a function set. Acts similar to a dataloader.

    Parameters
    ----------
    n_functions : int
        The number of functions that should be sampled when calling sample_functions.
    function_set : tp.domains.FunctionSet
        The function set from which functions should be sampled. Note that the size of the
        functions set needs to be larger or eqaul to n_functions.
    function_creation_interval : int, optional
        The interval at which new functions should be created. If set to 0, new functions are
        created every time sample_functions is called.
        The creation of new functions is handled by the function set. 
        The default value is 0.
    """
    def __init__(self, n_functions, function_set : FunctionSet, function_creation_interval : int = 0):
        self.n_functions = n_functions
        self.function_set = function_set
        self.function_creation_interval = function_creation_interval
        assert self.n_functions <= self.function_set.function_set_size, \
            "The number of functions to be sampled must be smaller than the function set size."

        self.iteration_counter = self.function_creation_interval
        self.current_indices = torch.zeros(n_functions, dtype=torch.int64)


    def _check_recreate_functions(self, device="cpu"):
        if self.iteration_counter >= self.function_creation_interval:
            self.function_set.create_functions(device=device)
            self.iteration_counter = -1
        self.iteration_counter += 1

    @abc.abstractmethod
    def sample_functions(self, device="cpu"):
        """ Sample functions from the function set.

        Parameters 
        ----------
        device : str, optional
            The device on which the functions should be stored. Default is cpu.

        Returns
        -------
        callable or torch.tensor
            Returns the sampled functions. If the function set is discrete, the functions
            can not be further evaluated and are therefore returned as a tensor. Otherwise
            a callable is returned that can be evaluated at any point.
        """
        raise NotImplementedError


class FunctionSamplerRandomUniform(FunctionSampler):
    """ Randomly samples functions from the function set.
    """
    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        self.current_indices = torch.randperm(self.function_set.function_set_size)[:self.n_functions] 
        return self.function_set.get_function(self.current_indices)


class FunctionSamplerOrdered(FunctionSampler):
    """ Samples functions in a ordered manner from the function set. When called
    will return the first n_functions functions from the function set and then increment
    the indices by n_functions. If the end of the function set is reached, the indices
    are reset to the beginning.
    """
    def __init__(self, n_functions, function_set : FunctionSet, function_creation_interval : int = 0):
        super().__init__(n_functions, function_set, function_creation_interval)
        self.current_indices = torch.arange(self.n_functions, dtype=torch.int64)

    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        current_out = self.function_set.get_function(self.current_indices)
        self.current_indices = (self.current_indices + self.n_functions) % self.function_set.function_set_size
        return current_out


class FunctionSamplerCoupled(FunctionSampler):
    """ A sampler that is coupled to another sampler, such that the same indices 
    of functions are sampled from both samplers.
    Can be usefull is two different data function sets are used where the data of 
    both sets is coupled and should therefore be samples accordingly.
    """
    def __init__(self, function_set : FunctionSet, coupled_sampler : FunctionSampler):
        super().__init__(coupled_sampler.n_functions, function_set, 
                         coupled_sampler.function_creation_interval)
        self.coupled_sampler = coupled_sampler

    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        
        self.current_indices = self.coupled_sampler.current_indices 
        return self.function_set.get_function(self.current_indices)