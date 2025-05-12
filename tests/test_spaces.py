import pytest
import torch
from collections import Counter, OrderedDict

from torchphysics.problem.spaces import (
        Space, R1, R2, R3, Rn, Z1, Z2, Z3, Zn, N1, N2, N3, Nn, FunctionSpace
    )

def test_create_space():
    s = Space({'x': 1})
    assert isinstance(s, Counter)
    assert isinstance(s, OrderedDict)


def test_product_of_spaces():
    s1 = Space({'x': 1})
    s2 = Space({'t': 3})
    s = s1 * s2
    assert s.dim == 4 


def test_space_contains_variable_name():
    s = Space({'x':1, 'y': 2})
    assert 'x' in s
    assert not 't' in s


def test_space_get_variable_dimension():
    s = Space({'x':1, 'y': 2})
    assert s['x'] == 1


def test_space_get_variable_dimension_for_list():
    s = Space({'x':1, 'y': 2, 't': 4})
    s2 = s[['x', 'y']]
    assert isinstance(s2, Space)
    assert 'x' in s2
    assert 'y' in s2
    assert not 't' in s2


def test_space_slice():
    s = Space({'x':1, 'y': 2})
    s2 = s[:'y']
    assert isinstance(s2, Space)
    assert 'x' in s2
    assert not 'y' in s2
    s2 = s['y':]
    assert isinstance(s2, Space)
    assert 'y' in s2
    assert not 'x' in s2


def test_space_contains_other_space():
    s = Space({'x':1, 'y': 2})
    s2 = Space({'x': 1})
    assert s2 in s


def test_space_doe_not_contain_other_objects():
    s = Space({'x':1, 'y': 2})
    assert not 5 in s


def test_space_get_variables():
    s = Space({'x':1, 'y': 2})
    assert 'x' in s.variables
    assert 'y' in s.variables


def test_space_serialize():
    s = Space({'x':1, 'y': 2})
    s_cls, s_dict = s.__reduce__()
    assert s_cls == Space
    assert isinstance(s_dict[0], OrderedDict)


def test_space_equal():
    x = R1('x')
    y = R1('y')
    z = R2('z')
    assert x == x
    assert not x == y
    assert not z == y
    assert x*y*z == x*y*z
    assert not x*y*z == x*z*y
    assert not y*x*z == x*y*z


def test_space_not_equal():
    x = R1('x')
    y = R1('y')
    z = R2('z')
    assert x != y
    assert not x != x
    assert z != y
    assert not x*y*z != x*y*z
    assert x*y*z != x*z*y
    assert y*x*z != x*y*z


def test_create_R1():
    r = R1('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 1


def test_create_R2():
    r = R2('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 2


def test_create_R3():
    r = R3('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 3

def test_create_Rn():
    r = Rn('x', 10)
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 10


def test_create_Z1():
    r = Z1('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 1


def test_create_Z2():
    r = Z2('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 2


def test_create_Z3():
    r = Z3('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 3

def test_create_Zn():
    r = Zn('x', 5)
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 5


def test_cast_tensor_into_integers():
    r = Z2('x')
    test_tensor = torch.tensor([[1.0, 0.2], [-2.3, 1.7], [3.0, 5]])
    casted_tensor = r.cast_tensor_into_space(test_tensor)
    expected_tensor = torch.tensor([[1.0, 0.0], [-2.0, 2.0], [3.0, 5]])
    assert torch.all(casted_tensor == expected_tensor)


def test_create_N1():
    r = N1('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 1


def test_create_N2():
    r = N2('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 2


def test_create_N3():
    r = N3('x')
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 3

def test_create_Nn():
    r = Nn('x', 134)
    assert isinstance(r, Counter)
    assert isinstance(r, OrderedDict)
    assert r.dim == 134

def test_cast_tensor_into_naturals():
    r = N2('x')
    test_tensor = torch.tensor([[1.0, 0.2], [-2.3, 1.7], [3.0, 5]])
    casted_tensor = r.cast_tensor_into_space(test_tensor)
    expected_tensor = torch.tensor([[1.0, 0.0], [2.0, 2.0], [3.0, 5]])
    assert torch.all(casted_tensor == expected_tensor)

def test_functionspace():
    X = R1('x')
    Y = R2('y')
    Z = FunctionSpace(X, Y)
    assert Z.input_space == X
    assert Z.output_space == Y


def test_functionspace_product():
    X = R1('x')
    Y = R2('y')
    U = R1("u")
    Z1 = FunctionSpace(X, Y)
    Z2 = FunctionSpace(X, U)
    Z = Z1 * Z2
    assert Z.input_space == X
    assert Z.output_space == Y*U


def test_functionspace_product_different_input():
    X = R1('x')
    T = R2("t")
    Y = R2('y')
    U = R1("u")
    Z1 = FunctionSpace(X, Y)
    Z2 = FunctionSpace(T, U)
    Z = Z1 * Z2
    assert Z.input_space == X*T
    assert Z.output_space == Y*U


def test_tensor_in_space_check():
    X = R1("x")
    test_tensor = torch.tensor([[1], [2], [3]])
    X.check_values_in_space(test_tensor)

def test_tensor_in_space_check_2D():
    X = R2("x")
    test_tensor = torch.tensor([[1, 2], [2, 1], [3, 5]])
    X.check_values_in_space(test_tensor)

def test_tensor_not_in_space_check():
    X = R1("x")
    test_tensor = torch.tensor([[1, 2], [2, 1], [3, 5]])
    with pytest.raises(AssertionError):
        X.check_values_in_space(test_tensor)