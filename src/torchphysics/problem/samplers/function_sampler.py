import torch

from ..domains.functionsets import FunctionSet


class FunctionSampler:

    def __init__(self, n_functions, function_set : FunctionSet, function_creation_interval : int):
        self.n_functions = n_functions
        self.function_set = function_set
        self.function_creation_interval = function_creation_interval
        if self.n_functions > self.function_set.function_set_size:
            Warning(f"""Sampled number of functions is larger than the set size. 
                    The size of the function set will be increased to 
                    the number {n_functions}.""")
            self.function_set.function_set_size = n_functions

        self.iteration_counter = self.function_creation_interval
        self.current_indices = torch.zeros(n_functions, dtype=torch.int64)


    def _check_recreate_functions(self, device="cpu"):
        if self.iteration_counter >= self.function_creation_interval:
            self.function_set.create_functions(device=device)
            self.iteration_counter = -1
        self.iteration_counter += 1


    def sample_functions(self, device="cpu"):
        pass


class RandomUniformFunctionSampler(FunctionSampler):
    # Randomly picks functions from the set
    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        self.current_indices = torch.randperm(self.function_set.function_set_size)[:self.n_functions] 
        return self.function_set.get_function(self.current_indices)


class OrderedFunctionSampler(FunctionSampler):
    # Picks function in order 1, 2, 3, ....
    def __init__(self, n_functions, function_set : FunctionSet, function_creation_interval : int):
        super().__init__(n_functions, function_set, function_creation_interval)
        self.current_indices = torch.arange(self.n_functions, dtype=torch.int64)

    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        current_out = self.function_set.get_function(self.current_indices)
        self.current_indices = (self.current_indices + self.n_functions) % self.function_set.function_set_size
        return current_out


class CoupledFunctionSampler(FunctionSampler):
    # Is coupled to another sampler and takes the same indices
    def __init__(self, function_set : FunctionSet, coupled_sampler : FunctionSampler):
        super().__init__(coupled_sampler.n_functions, function_set, 
                         coupled_sampler.function_creation_interval)
        self.coupled_sampler = coupled_sampler

    def sample_functions(self, device="cpu"):
        self._check_recreate_functions(device=device)
        
        self.current_indices = self.coupled_sampler.current_indices 
        return self.function_set.get_function(self.current_indices)