import torch
import pytest


from torchphysics.problem.conditions import *
from torchphysics.problem.spaces import Points, R1, R2
from torchphysics.problem.domains import Interval
from torchphysics.problem.samplers import GridSampler, DataSampler
from torchphysics.utils import UserFunction, laplacian, PointsDataLoader
from torchphysics.models import Parameter


def helper_fn(x, D=0.0):
    return Points(x**2 + D, R1('u'))


def test_create_general_condition():
    cond = Condition(name='test', weight=2.0, track_gradients=False)
    assert cond.name == 'test'
    assert cond.weight == 2.0
    assert not cond.track_gradients
    assert isinstance(cond, torch.nn.Module)


def test_track_gradients():
    cond = Condition()
    p = Points(torch.tensor([[2, 3.0, 0.0], [1, 1, 1]]), R1('t')*R2('x'))
    point_dict, new_points = p.track_coord_gradients()
    assert isinstance(point_dict, dict)
    assert isinstance(new_points, Points)
    assert Points.requires_grad
    assert 'x' in point_dict.keys()
    assert 't' in point_dict.keys()


def test_setup_data_functions():
    cond = Condition()
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=20)
    data_fn = {'f1': lambda x: x, 'f2': UserFunction(lambda x: 2*x)}
    changed_data_fn = cond._setup_data_functions(data_fn, ps)
    assert isinstance(changed_data_fn, dict)
    assert isinstance(changed_data_fn['f2'], UserFunction)
    assert isinstance(changed_data_fn['f1'], UserFunction)


def test_setup_data_functions_with_static_sampler():
    cond = Condition()
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=20).make_static()
    data_fn = {'f1': lambda x: x}
    changed_data_fn = cond._setup_data_functions(data_fn, ps)
    assert isinstance(changed_data_fn, dict)
    assert isinstance(changed_data_fn['f1'], UserFunction)
    assert torch.equal(changed_data_fn['f1'](), ps.sample_points())


def test_periodiccondition():
    module = UserFunction(helper_fn)
    interval = Interval(R1('x'), 0, 1)
    sampler = GridSampler(Interval(R1('y'), 0, 1), n_points=10).make_static()
    cond = PeriodicCondition(module,
                             interval,
                             lambda u_left, u_right: u_left-u_right,
                             non_periodic_sampler=sampler
                             )
    assert isinstance(cond, torch.nn.Module)
    assert cond.name == 'periodiccondition'
    assert cond.module == module
    out = cond()
    assert out == 1.0


def test_periodiccondition_data_fun_empty_sampler():
    def data_fun(x):
        return x**2
    module = UserFunction(helper_fn)
    interval = Interval(R1('x'), 0, 1)
    cond = PeriodicCondition(module,
                             interval,
                             lambda u_right, d_right: d_right-u_right,
                             data_functions={'d': data_fun}
                             )
    out = cond()
    assert out == 0.0


def test_create_datacondition():
    module = UserFunction(helper_fn)
    loader = PointsDataLoader((Points(torch.tensor([[0.0], [2.0]]), R1('x')),
                               Points(torch.tensor([[0.0], [4.0]]), R1('u'))),
                              batch_size=1)
    cond = DataCondition(module=module, dataloader=loader, norm=2)
    assert isinstance(cond, torch.nn.Module)
    assert cond.name == 'datacondition'
    assert cond.module == module
    assert next(iter(cond.dataloader))[0] == Points(torch.tensor([[0.0]]), R1('x'))


def test_datacondition_forward():
    module = UserFunction(helper_fn)
    loader = PointsDataLoader((Points(torch.tensor([[0.0], [2.0]]), R1('x')),
                               Points(torch.tensor([[0.0], [4.0]]), R1('u'))),
                              batch_size=1)
    cond = DataCondition(module=module, dataloader=loader, norm=2)
    out = cond()
    assert out == 0.0


def test_datacondition_forward_2():
    module = UserFunction(helper_fn)
    loader = PointsDataLoader((Points(torch.tensor([[0.0], [2.0]]), R1('x')),
                               Points(torch.tensor([[0.0], [1.0]]), R1('u'))),
                              batch_size=1)
    cond = DataCondition(module=module, dataloader=loader,
                         norm=2, use_full_dataset=True)
    out = cond()
    assert out == 4.5


def test_create_pinncondition():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = PINNCondition(module=module, sampler=ps, residual_fn=lambda u: u)
    assert isinstance(cond, torch.nn.Module)
    assert cond.name == 'pinncondition'
    assert cond.module == module
    assert cond.sampler == ps
    assert isinstance(cond.residual_fn, UserFunction)


def test_pinncondition_forward():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = PINNCondition(module=module, sampler=ps, residual_fn=lambda u: u)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_pinncondition_forward_with_2D_output():
    def module_fn(x):
        return Points(torch.column_stack((x, x+1)), R2('u'))
    module = UserFunction(module_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = PINNCondition(module=module, sampler=ps, residual_fn=lambda u: u)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    assert out.shape == torch.Size([])


def test_pinncondition_forward_with_derivative():
    def res_fn(u, x):
        return laplacian(u, x)
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=10)
    cond = PINNCondition(module=module, sampler=ps, residual_fn=res_fn)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    assert out.shape == torch.Size([])


def test_pinncondition_forward_with_parameter():
    module = UserFunction(helper_fn)
    param = Parameter(init=2.0, space=R1('D'))
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = PINNCondition(module=module, sampler=ps, residual_fn=lambda u: u,
                         parameter=param)
    out = cond()
    assert cond.parameter == param
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_pinncondition_forward_with_data_function():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    data_fn = {'f': lambda x: x}
    cond = PINNCondition(module=module, sampler=ps, residual_fn=lambda f, u: f+u,
                         data_functions=data_fn)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_create_ritzcondition():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = DeepRitzCondition(module=module, sampler=ps, integrand_fn=lambda u: u)
    assert isinstance(cond, torch.nn.Module)
    assert cond.name == 'deepritzcondition'
    assert cond.module == module
    assert cond.sampler == ps
    assert isinstance(cond.residual_fn, UserFunction)


def test_ritzcondition_forward():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    cond = DeepRitzCondition(module=module, sampler=ps,
                             integrand_fn=lambda u: torch.sum(u, dim=1))
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    assert torch.isclose(out, torch.tensor(0.3269), atol=0.0002)

def test_create_adaptiveweightscondition():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25).make_static()
    cond = AdaptiveWeightsCondition(module, ps, lambda u: u)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad

def test_ritzcondition_forward_with_data_function():
    module = UserFunction(helper_fn)
    ps = GridSampler(Interval(R1('x'), 0, 1), n_points=25)
    data_fn = {'f': lambda x: x}
    cond = DeepRitzCondition(module=module, sampler=ps,
                             integrand_fn=lambda u, f: torch.sum(u+f, dim=1),
                             data_functions=data_fn)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_create_parameter_condition():
    param = Parameter(init=2.0, space=R1('D'))
    def penalty(D): return D-3
    cond = ParameterCondition(parameter=param, penalty=penalty, weight=1)
    assert cond.parameter == param
    assert isinstance(cond.penalty, UserFunction)
    assert cond.name == 'parametercondition'
    assert not cond.track_gradients


def test_parameter_condition_forward():
    param = Parameter(init=2.0, space=R1('D'))
    def penalty(D): return D-3
    cond = ParameterCondition(parameter=param, penalty=penalty, weight=1)
    out = cond()
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    assert out == -1.0
