import torch
import pytest

from torchphysics.problem.conditions import *
from torchphysics.problem.spaces import R1, FunctionSpace
from torchphysics.problem.domains import Interval, FunctionSet, CustomFunctionSet
from torchphysics.problem.domains.functionsets.functionset import FunctionSetCollection
from torchphysics.problem.samplers import GridSampler
from torchphysics.utils.user_fun import UserFunction


def create_inputs():
    # Parameter space
    K = R1('k')
    I_k = Interval(K, 0, 4)
    # Function space
    T = R1('t')
    I_t = Interval(T, 0, 4)
    fn_space = FunctionSpace(I_t, R1('f'))
    return fn_space, I_k


def test_create_function_set():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    assert fn_set.function_space == fn_space
    assert fn_set.parameter_sampler == p_sampler
    assert fn_set.current_iteration_num == -1
    assert fn_set.param_batch is None


def test_len_of_function_set():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    assert len(fn_set) == len(p_sampler)


def test_function_set_sample_params():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    fn_set.sample_params()
    assert len(fn_set.param_batch) == 500


def test_create_meshgrid():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 50)
    fn_set = FunctionSet(fn_space, p_sampler) 
    t_sampler = GridSampler(fn_space.input_domain, 10)
    fn_set.sample_params()
    meshgrid = fn_set._create_meshgrid(t_sampler.sample_points())
    assert meshgrid.shape == (50, 10)
    assert 'k' in meshgrid.space
    assert 't' in meshgrid.space
    # check correct shape and distribution
    tensor = meshgrid.as_tensor
    assert tensor.shape == (50, 10, 2)
    for i in range(1, 10):
        assert tensor[0, 0, 0] == tensor[0, i, 0]


def test_add_functions_sets():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    p_sampler2 = GridSampler(I_k, 50)
    fn_set2 = FunctionSet(fn_space, p_sampler2)
    assert isinstance(fn_set + fn_set2, FunctionSetCollection)


def test_can_not_add_set_in_different_spaces():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    p_sampler2 = GridSampler(I_k, 50)
    fn_set2 = FunctionSet(FunctionSpace(I_k, R1('w')), p_sampler2)
    with pytest.raises(AssertionError):
        fn_set2 + fn_set


def test_multiple_add():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    p_sampler2 = GridSampler(I_k, 50)
    fn_set2 = FunctionSet(fn_space, p_sampler2)  
    fn_set3 = FunctionSet(fn_space, p_sampler2) 
    fn_collect = (fn_set + fn_set2) + fn_set3
    assert len(fn_collect.collection) == 3
    fn_collect = fn_set3 + (fn_set + fn_set2)
    assert len(fn_collect.collection) == 3


def test_len_of_function_set_collection():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    p_sampler2 = GridSampler(I_k, 50)
    fn_set2 = FunctionSet(fn_space, p_sampler2)  
    fn_set3 = FunctionSet(fn_space, p_sampler2) 
    fn_collect = fn_set + fn_set2
    assert len(fn_collect) == 550
    fn_collect += fn_set3
    assert len(fn_collect) == 600
    fn_collect += (fn_set3 + fn_set)
    assert len(fn_collect) == 1150


def test_sample_params_in_collection():
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = FunctionSet(fn_space, p_sampler)
    p_sampler2 = GridSampler(I_k, 50)
    fn_set2 = FunctionSet(fn_space, p_sampler2)  
    fn_collect = fn_set + fn_set2
    fn_collect.sample_params()
    assert len(fn_set.param_batch) == 500
    assert len(fn_set2.param_batch) == 50


def test_custom_function_set():
    def f(k, t):
        return k*t
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = CustomFunctionSet(fn_space, p_sampler, f)
    assert fn_set.function_space == fn_space
    assert fn_set.parameter_sampler == p_sampler
    assert fn_set.current_iteration_num == -1
    assert fn_set.param_batch is None
    assert isinstance(fn_set.custom_fn, UserFunction)


def test_evaluate_custom_function_set():
    def f(k, t):
        return k*t
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = CustomFunctionSet(fn_space, p_sampler, f)
    fn_set.sample_params()
    t_sampler = GridSampler(fn_space.input_domain, 10)
    fn_batch = fn_set.create_function_batch(t_sampler.sample_points()).as_tensor
    assert fn_batch.shape == (500, 10, 1)


def test_evaluate_custom_function_set_collection():
    def f(k, t):
        return k*t
    fn_space, I_k = create_inputs()
    p_sampler = GridSampler(I_k, 500)
    fn_set = CustomFunctionSet(fn_space, p_sampler, f)
    def f2(k, t):
        return k*torch.sin(t**2)
    p_sampler2 = GridSampler(I_k, 20)  
    fn_set += CustomFunctionSet(fn_space, p_sampler2, UserFunction(f2))
    fn_set.sample_params()
    t_sampler = GridSampler(fn_space.input_domain, 22)
    fn_batch = fn_set.create_function_batch(t_sampler.sample_points()).as_tensor
    assert fn_batch.shape == (520, 22, 1)