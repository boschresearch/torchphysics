import torch
import pytest

from torchphysics.problem.domains.functionsets import FunctionSet
from torchphysics.problem.spaces import R1, R2, FunctionSpace, Points

def test_create_fn_set():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    assert fn_set.function_space == space
    assert fn_set.function_set_size == 100


def test_fn_set_not_discrete():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    assert not fn_set.is_discretized


def test_fn_set_discretize():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    points = Points(torch.rand((1, 100, 1)), R1("t"))
    _ = fn_set.discretize(points)


def test_fn_set_discretization_of_default_wrong():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    assert not fn_set.is_discretization_of(None)


def test_fn_set_discretization_of():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    points = Points(torch.rand((1, 100, 1)), R1("t"))
    discrete_fn_set = fn_set.discretize(points)
    assert discrete_fn_set.is_discretization_of(fn_set)


def test_discrete_fn_set_is_discrete():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    points = Points(torch.rand((1, 100, 1)), R1("t"))
    discrete_fn_set = fn_set.discretize(points)
    assert discrete_fn_set.is_discretized


def test_fn_set_product():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("y"))
    fn_set2 = FunctionSet(space2, 100)
    new_fn_set = fn_set * fn_set2
    assert new_fn_set.function_space.output_space == R2("x")*R2("y")
    assert new_fn_set.function_space.input_space == R1("t")


def test_fn_set_product_different_input():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("x"), R2("y"))
    fn_set2 = FunctionSet(space2, 100)
    new_fn_set = fn_set * fn_set2
    assert new_fn_set.function_space.output_space == R2("x")*R2("y")
    assert new_fn_set.function_space.input_space == R1("t")*R1("x")


def test_fn_set_product_same_output():
    space = FunctionSpace(R1("t"), R2("u"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("x"), R2("u"))
    fn_set2 = FunctionSet(space2, 100)
    with pytest.raises(AssertionError):
        _ = fn_set * fn_set2


def test_fn_set_product_three_times():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("y"))
    fn_set2 = FunctionSet(space2, 100)
    space3 = FunctionSpace(R1("t"), R1("z"))
    fn_set3 = FunctionSet(space3, 100)
    new_fn_set = fn_set * fn_set2
    new_fn_set = new_fn_set*fn_set3
    assert new_fn_set.function_space.output_space == R2("x")*R2("y")*R1("z")
    assert new_fn_set.function_space.input_space == R1("t")
    assert new_fn_set.function_set_size == 100
    
    new_fn_set = fn_set * fn_set2
    new_fn_set = fn_set3*new_fn_set
    assert new_fn_set.function_space.output_space == R2("x")*R2("y")*R1("z")
    assert new_fn_set.function_space.input_space == R1("t")
    assert new_fn_set.function_set_size == 100


def test_fn_set_product_with_product():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("y"))
    fn_set2 = FunctionSet(space2, 100)
    space3 = FunctionSpace(R1("t"), R1("z"))
    fn_set3 = FunctionSet(space3, 100)
    space4 = FunctionSpace(R1("w"), R1("f"))
    fn_set4 = FunctionSet(space4, 100)

    new_fn_set = fn_set * fn_set2
    new_fn_set2 = fn_set3 * fn_set4
    new_fn_set *= new_fn_set2
    assert new_fn_set.function_space.output_space == R2("x")*R2("y")*R1("z")*R1("f")
    assert new_fn_set.function_space.input_space == R1("t")*R1("w")
    assert new_fn_set.function_set_size == 100

    new_fn_set = fn_set * fn_set2
    new_fn_set2 = fn_set3 * fn_set4
    new_fn_set2 *= new_fn_set
    assert new_fn_set2.function_space.output_space == R1("z")*R1("f")*R2("x")*R2("y")
    assert new_fn_set2.function_space.input_space == R1("t")*R1("w")
    assert new_fn_set2.function_set_size == 100


def test_fn_set_append_wrong_output_space():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("y"))
    fn_set2 = FunctionSet(space2, 100)
    with pytest.raises(AssertionError):
        _ = fn_set.append(fn_set2)


def test_fn_set_append():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("x"))
    fn_set2 = FunctionSet(space2, 100)
    new_fn_set = fn_set.append(fn_set2)
    assert new_fn_set.function_space.output_space == R2("x")
    assert new_fn_set.function_set_size == 200


def test_fn_set_append_two_times():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("x"))
    fn_set2 = FunctionSet(space2, 100)
    space3 = FunctionSpace(R1("t"), R2("x"))
    fn_set3 = FunctionSet(space3, 100)
    new_fn_set = fn_set.append(fn_set2).append(fn_set3)
    assert new_fn_set.function_space.output_space == R2("x")
    assert new_fn_set.function_set_size == 300


def test_fn_set_append_two_combies():
    space = FunctionSpace(R1("t"), R2("x"))
    fn_set = FunctionSet(space, 100)
    space2 = FunctionSpace(R1("t"), R2("x"))
    fn_set2 = FunctionSet(space2, 100)
    space3 = FunctionSpace(R1("t"), R2("x"))
    fn_set3 = FunctionSet(space3, 100)
    space4 = FunctionSpace(R1("t"), R2("x"))
    fn_set4 = FunctionSet(space4, 100)

    new_fn_set = fn_set.append(fn_set2)
    new_fn_set2 = fn_set4.append(fn_set3)
    new_fn_set = new_fn_set.append(new_fn_set2)
    assert new_fn_set.function_space.output_space == R2("x")
    assert new_fn_set.function_space.input_space == R1("t")
    assert new_fn_set.function_set_size == 400