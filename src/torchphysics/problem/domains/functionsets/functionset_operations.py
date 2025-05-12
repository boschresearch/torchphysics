import torch
import abc

from ...spaces.points import Points
from .functionset import FunctionSet, DiscretizedFunctionSet
from ....utils.user_fun import UserFunction


class FunctionSetProduct(FunctionSet):
    """The product of multiple function sets.

    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace    
        The function space of the set.
    function_sets : list
        A list of FunctionSets that should be multiplied together.
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
        assert self.function_set_size == other.function_set_size, \
                """Both FunctionSets need the same size!"""
        
        if isinstance(other, DiscretizedFunctionSet):
            if not self.is_discretized:
                raise ValueError("Other FunctionSet is discrete but this set is continuous.")
            else:
                assert torch.equal(self.function_sets[0].locations, other.locations), \
                """Both DiscretizedFunctionSets need the same locations for creating the product!"""
                return FunctionSetProduct(self.function_space*other.function_space, 
                                            self.function_sets + [other])
        elif isinstance(other, FunctionSetProduct):
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
        return FunctionSetProduct(self.function_space, 
                                  [f.discretize(locations) for f in self.function_sets])


    def _evaluate_product(self, locations):
        point_list = [fn(locations) for fn in self.fn_list]
        return Points.joined(*point_list)


class FunctionSetCollection(FunctionSet):
    """Collection of multiple FunctionSets. Used for combining different kinds of 
    functions into one single set.

    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace    
        The function space of the set.
    function_sets : list
        A list of FunctionSets that should be combined.
    """
    def __init__(self, function_space, function_sets):
        set_size = 0
        for fn_set in function_sets:
            set_size += fn_set.function_set_size

        super().__init__(function_space, set_size)
        
        self.function_sets = function_sets
        self.current_idx = []
        self.device = "cpu"

    @property
    def is_discretized(self):
        for fn_set in self.function_sets:
            if fn_set.is_discretized:
                return True
        return False
    
    def is_discretization_of(self, function_set):
        if isinstance(function_set, FunctionSetCollection):
            for self_set in self.function_sets:
                for other_set in function_set.function_sets:
                    if self_set.is_discretization_of(other_set):
                        break
                else:
                    return False
            return True
        return False
    
    def create_functions(self, device="cpu"):
        for fn_set in self.function_sets:
            fn_set.create_functions(device)
        self.device = device

    def get_function(self, idx):
        if isinstance(idx, int): idx = [idx]
        self.current_idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        return self._evaluate_collection

    def _evaluate_collection(self, location : Points):
        # We go over each function set of the collection and check
        # if the given indices belong to this set to sample functions there.
        final_output = []
        # we may need to fix the order at the end such that the input index
        # also fits the output values 
        idx_permut = []
        size_counter = 0
        for fn_set in self.function_sets:
            valid_idx = torch.where(
                (self.current_idx >= size_counter) & (self.current_idx < size_counter + fn_set.function_set_size)
                )[0]
            if len(valid_idx) > 0:
                fn_set_idx = self.current_idx[valid_idx]
                fn_set_idx -= size_counter # shift index back to start at zero
                fn_set_output = fn_set.get_function(fn_set_idx)(location).as_tensor
                if len(final_output) == 0:
                    final_output = fn_set_output
                    idx_permut = valid_idx
                else:
                    final_output = torch.cat((final_output, fn_set_output), dim=0)
                    idx_permut = torch.cat((idx_permut, valid_idx))
            
            size_counter += fn_set.function_set_size
        
        final_output = final_output[idx_permut]
        return Points(final_output, self.function_space.output_space)

    def discretize(self, locations):
        return FunctionSetCollection(self.function_space, 
                                  [f.discretize(locations) for f in self.function_sets])
    
    def append(self, other):
        assert self.function_space.output_space == other.function_space.output_space, \
                """Both FunctionSets need the same output space!"""
        if isinstance(other, FunctionSetCollection):
            return FunctionSetCollection(self.function_space, 
                                    self.function_sets + other.function_sets)
        else:
            return FunctionSetCollection(self.function_space, 
                                    self.function_sets + [other])
        

class FunctionSetArithmetics(FunctionSet):

    def __init__(self, function_space, function_sets):
        for i in range(len(function_sets)):
            assert function_sets[0].function_set_size == function_sets[i].function_set_size, \
                "Size of function sets is not the same, you can not combine them!"
            assert function_sets[0].is_discretized == function_sets[i].is_discretized, \
                "Some function sets are discretized some are not, you can not combine them!"

        super().__init__(function_space, function_sets[0].function_set_size)
        
        self.function_sets = function_sets
        self.current_idx = []

    @property
    def is_discretized(self):
        return self.function_sets[0].is_discretized
    
    def is_discretization_of(self, function_set):
        if isinstance(function_set, FunctionSetArithmetics):
            for self_set in self.function_sets:
                for other_set in function_set.function_sets:
                    if self_set.is_discretization_of(other_set):
                        break
                else:
                    return False
            return True
        return False
    
    def create_functions(self, device="cpu"):
        for fn_set in self.function_sets:
            fn_set.create_functions(device)
    
    def get_function(self, idx):
        if self.is_discretized:
            data_list = [fn_set.get_function(idx).as_tensor for fn_set in self.function_sets]
            return self.arithmetic_function(data_list)
        else:
            self.fn_list = [fn_set.get_function(idx) for fn_set in self.function_sets]
            return self._evaluate_product

    def _evaluate_product(self, locations):
        data_list = [fn(locations).as_tensor for fn in self.fn_list]
        return self.arithmetic_function(data_list)

    @abc.abstractmethod
    def arithmetic_function(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def discretize(self, locations):
        raise NotImplementedError


class FunctionSetAdd(FunctionSetArithmetics):
    """
    A class handling the pointwise addition of two sets.
    """
    def arithmetic_function(self, data):
        output = torch.zeros_like(data[0])
        for i in range(len(data)):
            output += data[i]
        return Points(output, self.function_space.output_space)

    def discretize(self, locations):
        return FunctionSetAdd(self.function_space, 
                    [f.discretize(locations) for f in self.function_sets])
    
    def __add__(self, other):
        if isinstance(other, FunctionSetAdd):
            return FunctionSetAdd(self.function_space, 
                                    self.function_sets + other.function_sets)
        else:
            return FunctionSetAdd(self.function_space, 
                                    self.function_sets + [other])
        

class FunctionSetSubstract(FunctionSetArithmetics):
    """
    A class handling the pointwise substraction of two sets.
    """
    def arithmetic_function(self, data):
        # data will be always only two different function sets
        output = data[1] - data[0]
        return Points(output, self.function_space.output_space)

    def discretize(self, locations):
        return FunctionSetSubstract(self.function_space, 
                    [f.discretize(locations) for f in self.function_sets])
    

class FunctionSetTransform(FunctionSet):
    """A class that acts as a wrapper of a different function set to further modify 
    the created functions. E.g., clamping the values produced by a different
    function set.

    Parameters
    ----------
    function_set : tp.domains.FunctionSet
        The function set that should be transformed.
    transformation : callable
        The function that carries out the transformation. This transformation will
        be carried out "pointwise".
    
    """
    def __init__(self, function_set : FunctionSet, transformation):
        self.fn_set = function_set
        self.transformation = UserFunction(transformation)
        super().__init__(self.fn_set.function_space, self.fn_set.function_set_size)

    @property
    def is_discretized(self):
        return self.fn_set.is_discretized
    
    def is_discretization_of(self, function_set):
        if isinstance(function_set, FunctionSetTransform):
            return self.fn_set.is_discretization_of(function_set.fn_set)
        else:
            return self.fn_set.is_discretization_of(function_set)
    
    def create_functions(self, device="cpu"):
        self.fn_set.create_functions(device=device)

    def get_function(self, idx):
        if self.fn_set.is_discretized:
            transformed_pnt = self.transformation(self.fn_set.get_function(idx))
            return Points(transformed_pnt, self.function_space.output_space)
        else:
            self.fn_set_eval = self.fn_set.get_function(idx)
            return self._eval_fn_set

    def _eval_fn_set(self, locations):
        transformed_pnt = self.transformation(self.fn_set_eval(locations))
        return Points(transformed_pnt, self.function_space.output_space)

    def discretize(self, locations):
        return FunctionSetTransform(self.fn_set.discretize(locations), self.transformation)