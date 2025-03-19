import torch
import pytest

from torchphysics.problem.spaces import R1, FunctionSpace
from torchphysics.problem.domains.functionsets import CustomFunctionSet
from torchphysics.problem.domains.functionsets.functionset import FunctionSetProduct, FunctionSetCollection
from torchphysics.problem.samplers import GridSampler
from torchphysics.problem.domains.domain1D import Interval
from torchphysics.problem.spaces.points import Points
from torchphysics.utils.user_fun import UserFunction

def test_create_custom_fn_set():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    assert fn_set.function_set_size == 100
    assert fn_set.parameter_sampler == sampler
    assert fn_set.custom_fn.fun == custom_fn


def test_create_custom_fn_set_with_user_fn():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, UserFunction(custom_fn))
    assert fn_set.function_set_size == 100
    assert fn_set.parameter_sampler == sampler
    assert fn_set.custom_fn.fun == custom_fn


def test_custom_fn_set_create_functions():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set.create_functions()
    assert torch.is_tensor(fn_set.param_samples.as_tensor)
    assert fn_set.param_samples.as_tensor.shape[0] == 100
    assert fn_set.param_samples.as_tensor.shape[1] == 1


def test_custom_fn_set_get_single_fn():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    fisrt_value = sampler.sample_points().as_tensor[0]
    custom_fn = lambda k,x : k + 2 + x
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set.create_functions()
    fn_0 = fn_set.get_function(0)
    assert callable(fn_0)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    fn_out = fn_0(input_points)
    assert fn_out.space == R1("u")
    assert torch.all(fn_out.as_tensor == input_points.as_tensor + 2.0 + fisrt_value)


def test_custom_fn_set_get_multiple_fn():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100).make_static()
    custom_fn = lambda k,x : k + 2 + x
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set.create_functions()
    fn_0 = fn_set.get_function([0, 1, 2, 3])
    assert callable(fn_0)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    fn_out = fn_0(input_points)
    assert fn_out.space == R1("u")
    assert fn_out.as_tensor.shape[0] == 4
    assert fn_out.as_tensor.shape[1] == 10
    for i in range(4):
        fisrt_value = sampler.sample_points().as_tensor[i]
        assert torch.all(torch.isclose(fn_out.as_tensor[i:i+1], input_points.as_tensor + 2.0 + fisrt_value))


def test_custom_fn_set_get_multiple_fn_with_different_input():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100).make_static()
    custom_fn = lambda k,x : k + 2 + x
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set.create_functions()
    fn_0 = fn_set.get_function([0, 1, 2, 3])
    assert callable(fn_0)
    input_points = Points(torch.rand((4, 10, 1)), R1("x"))
    fn_out = fn_0(input_points)
    assert fn_out.space == R1("u")
    assert fn_out.as_tensor.shape[0] == 4
    assert fn_out.as_tensor.shape[1] == 10
    for i in range(4):
        fisrt_value = sampler.sample_points().as_tensor[i]
        assert torch.all(torch.isclose(fn_out.as_tensor[i:i+1], input_points[i:i+1].as_tensor + 2.0 + fisrt_value))


def test_discretize_custom_fn_set():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn_set = fn_set.discretize(input_points)
    assert discrete_fn_set.is_discretized
    assert discrete_fn_set.is_discretization_of(fn_set)


def test_discretize_custom_fn_set_get_fn():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn_set = fn_set.discretize(input_points)
    discrete_fn_set.create_functions()
    fn_set_out = discrete_fn_set.get_function(0)
    assert not callable(fn_set_out)
    assert fn_set_out.as_tensor.shape[0] == 1


def test_discretize_custom_fn_set_get_fn_multiple():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn_set = fn_set.discretize(input_points)
    discrete_fn_set.create_functions()
    fn_set_out = discrete_fn_set.get_function([0, 1, 2])
    assert not callable(fn_set_out)
    assert fn_set_out.as_tensor.shape[0] == 3


def test_custom_fn_set_create_product():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    fn_set *= fn_set2
    fn_set.create_functions()


def test_custom_fn_set_product_get_function():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    fn_set *= fn_set2
    fn_set.create_functions()
    fn = fn_set.get_function([0, 1, 2])
    assert callable(fn)
    assert fn_set.function_sets[0].current_idx == [0, 1, 2]
    assert fn_set.function_sets[1].current_idx == [0, 1, 2]


def test_custom_fn_set_product_evaluate_function():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    fn_set *= fn_set2
    fn_set.create_functions()
    fn = fn_set.get_function([0, 1, 2])
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    out = fn(input_points)
    assert out.as_tensor.shape[0] == 3
    assert out.as_tensor.shape[1] == 10
    assert out.as_tensor.shape[2] == 2


def test_custom_fn_set_create_append():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    fn_set.create_functions()


def test_custom_fn_set_append_get_function():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    fn_set.create_functions()
    fn = fn_set.get_function([0, 1, 2])
    assert callable(fn)
    assert all(fn_set.current_idx == torch.tensor([0, 1, 2]))


def test_custom_fn_set_append_evaluate_function_first():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    fn_set.create_functions()
    fn = fn_set.get_function([0, 1, 2])
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    out = fn(input_points)
    assert out.as_tensor.shape[0] == 3
    assert out.as_tensor.shape[1] == 10
    assert out.as_tensor.shape[2] == 1


def test_custom_fn_set_append_evaluate_function_both():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    fn_set.create_functions()
    fn = fn_set.get_function([0, 1, 122])
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    out = fn(input_points)
    assert out.as_tensor.shape[0] == 3
    assert out.as_tensor.shape[1] == 10
    assert out.as_tensor.shape[2] == 1


def test_custom_fn_set_discretize_product():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    fn_set *= fn_set2
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert isinstance(discrete_fn, FunctionSetProduct)


def test_custom_fn_set_discretize_product_check_discrete():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    fn_set *= fn_set2
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert discrete_fn.is_discretized
    assert not fn_set.is_discretized


def test_custom_fn_set_discretize_product_check_discretization_of():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    space2 = FunctionSpace(R1("x"), R1("y"))
    fn_set2 = CustomFunctionSet(space2, sampler, custom_fn)
    space3 = FunctionSpace(R1("x"), R1("z"))
    fn_set3 = CustomFunctionSet(space3, sampler, custom_fn)
    fn_set *= fn_set2
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert discrete_fn.is_discretization_of(fn_set)
    assert not discrete_fn.is_discretization_of(fn_set2)
    assert not discrete_fn.is_discretization_of(fn_set2*fn_set3)


def test_custom_fn_set_discretize_append():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert isinstance(discrete_fn, FunctionSetCollection)


def test_custom_fn_set_discretize_check_discrete():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert discrete_fn.is_discretized
    assert not fn_set.is_discretized


def test_custom_fn_set_discretize_check_discretization_of():
    sampler = GridSampler(Interval(R1("k"), 0, 1), 100)
    custom_fn = lambda k : k+2
    space = FunctionSpace(R1("x"), R1("u"))
    fn_set = CustomFunctionSet(space, sampler, custom_fn)
    fn_set2 = CustomFunctionSet(space, sampler, custom_fn)
    fn_set = fn_set.append(fn_set2)
    input_points = Points(torch.rand((1, 10, 1)), R1("x"))
    discrete_fn = fn_set.discretize(input_points)
    assert discrete_fn.is_discretization_of(fn_set)
    assert not discrete_fn.is_discretization_of(fn_set2)
    assert not discrete_fn.is_discretization_of(fn_set2.append(fn_set2))