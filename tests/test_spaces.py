from collections import Counter, OrderedDict

from torchphysics.problem.spaces import Space, R1, R2, R3, FunctionSpace
from torchphysics.problem.domains import Interval


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

def test_functionspace():
    X = R1('x')
    Y = R2('y')
    I = Interval(X, 0, 1)
    Z = FunctionSpace(I, Y)
    assert Z.input_space == X