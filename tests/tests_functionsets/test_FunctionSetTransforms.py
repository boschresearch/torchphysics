import torch
import pytest

from torchphysics.problem.domains.functionsets import FunctionSet
from torchphysics.problem.domains.functionsets.functionset_operations import (
        FunctionSetAdd, FunctionSetSubstract, FunctionSetTransform
    )
from torchphysics.problem.spaces import R1, R2, FunctionSpace, Points
from torchphysics.problem.domains.functionsets import CustomFunctionSet
from torchphysics.problem.samplers import GridSampler
from torchphysics.problem.domains.domain1D import Interval
from torchphysics.problem.spaces.points import Points
from torchphysics.utils.user_fun import UserFunction


def test_create_fn_set_sum():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    fn_set2 = FunctionSet(space, 100)
    fn_set += fn_set2
    assert isinstance(fn_set, FunctionSetAdd)
    assert fn_set.function_space == space
    assert fn_set.function_set_size == 100


def test_create_fn_set_sum_multiple():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    fn_set2 = FunctionSet(space, 100)
    fn_set += fn_set2
    fn_set3 = FunctionSet(space, 100)
    fn_set4 = fn_set + fn_set3
    assert isinstance(fn_set4, FunctionSetAdd)
    assert fn_set4.function_space == space
    assert fn_set4.function_set_size == 100

    fn_set4 = fn_set3 + fn_set
    assert isinstance(fn_set4, FunctionSetAdd)
    assert fn_set4.function_space == space
    assert fn_set4.function_set_size == 100

    fn_set5 = fn_set + fn_set4
    assert isinstance(fn_set5, FunctionSetAdd)
    assert fn_set5.function_space == space
    assert fn_set5.function_set_size == 100


def test_create_fn_set_substract():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    fn_set2 = FunctionSet(space, 100)
    fn_set -= fn_set2
    assert isinstance(fn_set, FunctionSetSubstract)
    assert fn_set.function_space == space
    assert fn_set.function_set_size == 100


def test_create_fn_set_sum_and_substract():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    fn_set2 = FunctionSet(space, 100)
    fn_set += fn_set2
    fn_set3 = FunctionSet(space, 100)
    fn_set4 = fn_set + fn_set3
    assert isinstance(fn_set4, FunctionSetAdd)
    assert fn_set4.function_space == space
    assert fn_set4.function_set_size == 100

    fn_set4 = fn_set3 - fn_set
    assert isinstance(fn_set4, FunctionSetSubstract)
    assert fn_set4.function_space == space
    assert fn_set4.function_set_size == 100


def make_fnsets():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)

    custom_fn2 = lambda k,x : torch.sin(k*x) + x*k
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn2)
    return fn_set, fn_set2

def test_fn_set_sum_discrete():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    assert not fn_set.is_discretized


def test_fn_set_sum_make_discrete():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    fn_set_discrete = fn_set.discretize(gridpoints)
    assert fn_set_discrete.is_discretized


def test_fn_set_substract_make_discrete():
    fn_set, fn_set2 = make_fnsets()
    fn_set -= fn_set2
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    fn_set_discrete = fn_set.discretize(gridpoints)
    assert fn_set_discrete.is_discretized


def test_fn_set_sum_discretization_of():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    fn_set_discrete = fn_set.discretize(gridpoints)
    assert fn_set_discrete.is_discretization_of(fn_set)
    assert not fn_set_discrete.is_discretization_of(fn_set2)


def test_fn_set_sum_create_fns():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    fn_set.create_functions()


def test_fn_set_sum_get_fns():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    fn_set.create_functions()
    eval_fn = fn_set.get_function([1, 2, 3])
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    assert eval_fn(gridpoints).as_tensor.shape == (3, 3, 1)


def test_fn_set_sum_get_fns_discrete():
    fn_set, fn_set2 = make_fnsets()
    fn_set += fn_set2
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    fn_set_discrete = fn_set.discretize(gridpoints)
    fn_set_discrete.create_functions()
    eval_fn = fn_set_discrete.get_function([1, 2, 3])
    assert eval_fn.as_tensor.shape == (3, 3, 1)


def test_fn_set_substract_get_fns():
    fn_set, fn_set2 = make_fnsets()
    fn_set -= fn_set2
    fn_set.create_functions()
    eval_fn = fn_set.get_function([1, 2, 3])
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    assert eval_fn(gridpoints).as_tensor.shape == (3, 3, 1)


def test_fn_set_transform_create():
    fn_set, _ = make_fnsets()
    transform_set = FunctionSetTransform(fn_set, lambda u : torch.clamp(u, 0.0, 0.1))
    assert transform_set.fn_set == fn_set
    assert callable(transform_set.transformation)


def test_fn_set_transform_discrete():
    fn_set, _ = make_fnsets()
    transform_set = FunctionSetTransform(fn_set, lambda u : torch.clamp(u, 0.0, 0.1))
    assert not transform_set.is_discretized
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    set_discrete = transform_set.discretize(gridpoints)
    assert set_discrete.is_discretization_of(transform_set)
    assert set_discrete.is_discretization_of(fn_set)


def test_fn_set_transform_evaluate_fn():
    fn_set, _ = make_fnsets()
    transform_set = FunctionSetTransform(fn_set, lambda u : torch.clamp(u, 0.0, 0.1))
    transform_set.create_functions()
    some_fns = transform_set.get_function([3, 4, 5, 6, 10])
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    output = some_fns(gridpoints).as_tensor
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 0.1)


def test_fn_set_transform_evaluate_fn_discrete():
    fn_set, _ = make_fnsets()
    transform_set = FunctionSetTransform(fn_set, lambda u : torch.clamp(u, 0.0, 0.1))
    gridpoints = Points(torch.tensor([0, 0.5, 1.0]).reshape(1, -1, 1), R1("x"))
    set_discrete = transform_set.discretize(gridpoints)
    set_discrete.create_functions()
    output = set_discrete.get_function([3, 4, 5, 6, 10]).as_tensor
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 0.1)