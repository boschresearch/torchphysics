import pytest
import torch

from torchphysics.problem.spaces.points import Points
from torchphysics.problem.spaces.space import R1
from torchphysics.utils.user_fun import UserFunction, DomainUserFunction


def _helper_fun(x, y, t):
    return x+y+t


def test_create_user_function():
    user_fn = UserFunction(_helper_fun)
    assert user_fn.fun == _helper_fun
    assert user_fn.necessary_args == ['x', 'y', 't']


def test_get_name_of_user_function():
    user_fn = UserFunction(_helper_fun)
    assert user_fn.__name__() == _helper_fun.__name__


def test_create_user_function_with_constant_value():
    user_fn = UserFunction([0, 1])
    assert user_fn.fun[0] == 0
    assert user_fn.fun[1] == 1
    assert user_fn.necessary_args == []
    assert user_fn.optional_args ==[]


def test_create_user_function_from_user_function():
    user_fn = UserFunction(_helper_fun)
    user_fn_2 = UserFunction(user_fn)
    assert user_fn.fun == user_fn_2.fun


def test_cant_create_user_fun_with_variable_args():
    def test_fn(*args):
        return 
    with pytest.raises(ValueError):
        _ = UserFunction(test_fn)


def test_create_user_fun_with_default_values():
    def test_fn(x, d=4):
        return x + d
    user_fn = UserFunction(test_fn)
    assert user_fn.args == ['x', 'd']
    assert user_fn.necessary_args == ['x']
    assert user_fn.optional_args == ['d']


def test_call_user_fun():
    user_fn = UserFunction(_helper_fun)
    assert user_fn({'x': 3, 'y': 2, 't': 1}) == 6


def test_call_user_fun_if_constant():
    user_fn = UserFunction(4.0)
    assert user_fn({'x': 3, 'y': 2, 't': 1}) == 4.0


def test_call_user_fun_with_point():
    user_fn = UserFunction(_helper_fun)
    p = Points(torch.tensor([[2, 2, 0.0]]), R1('x')*R1('y')*R1('t'))
    assert user_fn(p) == 4


def test_call_user_fun_with_missing_information():
    user_fn = UserFunction(_helper_fun)
    with pytest.raises(AssertionError):
        _ = user_fn({'x': 3})


def test_call_user_fun_with_default_values():
    def test_fn(x, d=4):
        return x + d
    user_fn = UserFunction(test_fn)
    assert user_fn({'x' : 3}) == 7
    assert user_fn({'d' : 3, 'x': 3}) == 6


def test_call_user_fun_with_apply_to_batch():
    user_fn = UserFunction(_helper_fun)
    p = Points(torch.ones(10, 3), R1('x')*R1('y')*R1('t'))
    out = user_fn(p, vectorize=True)
    assert len(out) == 10


def test_call_user_fun_with_apply_to_batch_if_some_value_is_None():
    def test_fn(x, y):
        if x==0:
            return None
        return x + y
    user_fn = UserFunction(test_fn)
    t = torch.ones(10, 1)
    t[0] = 0
    p = {'x': t, 'y': torch.tensor([[2.0]])}
    out = user_fn(p, vectorize=True)
    assert len(out) == 9


def test_partially_evaluate_if_constant():
    user_fn = UserFunction(3.3)
    assert user_fn.partially_evaluate(t=0.02) == 3.3


def test_partially_evaluate_if_all_inputs_are_given():
    user_fn = UserFunction(_helper_fun)
    assert user_fn.partially_evaluate(x=1, y=2, t=0) == 3


def test_partially_evaluate():
    user_fn = UserFunction(_helper_fun)
    part_fn = user_fn.partially_evaluate(x=1, y=2)
    assert isinstance(part_fn, UserFunction)
    assert part_fn.necessary_args == ['t']
    assert part_fn.optional_args == ['x', 'y']
    assert part_fn({'t': 1.0}) == 4


def test_set_user_function_default():
    user_fn = UserFunction(_helper_fun)
    user_fn.set_default(x=3)
    assert user_fn.optional_args == ['x']
    assert user_fn.defaults['x'] == 3


def test_user_function_remove_default_after_set():
    user_fn = UserFunction(_helper_fun)
    user_fn.set_default(x=3)
    user_fn.remove_default(x=3)
    assert user_fn.optional_args == []


def test_user_function_remove_original_default():
    def test_fn(x, d=3):
        return x + d
    user_fn = UserFunction(test_fn)
    assert user_fn.optional_args == ['d']
    user_fn.remove_default('d')
    assert user_fn.optional_args == []


def test_create_domain_user_function():
    user_fn = DomainUserFunction(_helper_fun)
    assert user_fn.fun == _helper_fun
    assert user_fn.necessary_args == ['x', 'y', 't']


def test_create_domain_user_function_from_user_function():
    user_fn = UserFunction(_helper_fun)
    user_fn = DomainUserFunction(user_fn)
    assert user_fn.fun == _helper_fun
    assert user_fn.necessary_args == ['x', 'y', 't']


def test_call_domain_user_function_if_constant_and_not_tensor():
    user_fn = DomainUserFunction(3.0)
    assert isinstance(user_fn(), torch.Tensor)
    assert user_fn() == 3.0


def test_call_domain_user_function_if_constant_and_tensor():
    user_fn = DomainUserFunction(torch.tensor([2, 0.0]))
    assert torch.equal(user_fn(), torch.tensor([2, 0.0]))


def test_call_domain_user_function_if_callable_and_output_is_tensor():
    user_fn = DomainUserFunction(_helper_fun)
    p = Points(torch.ones(10, 3), R1('x')*R1('y')*R1('t'))
    out = user_fn(p)
    assert out.shape == (10, 1, 1)


def test_call_domain_user_function_if_callable_and_output_is_not_tensor():
    def test_fn(x):
        out = [[x, x]]
        return out
    user_fn = DomainUserFunction(test_fn)
    out = user_fn({'x': torch.tensor(2)})
    assert out.shape == (1, 1, 2)
    assert isinstance(out, torch.Tensor)