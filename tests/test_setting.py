import pytest
import torch
import numpy as np

from torchphysics.problem.variables import Variable
from torchphysics.problem.parameters import Parameter
from torchphysics.problem.condition import (DiffEqBoundaryCondition, 
                                            DiffEqCondition)
from torchphysics.problem.domain import Interval
import torchphysics.setting as setting


# Test SimpleDataset (helper class)
def test_create_simple_dataset():
    s = setting.SimpleDataset(data=np.array([0, 3, 4]), 
                              iterations=20)
    assert np.allclose(s.data, [0, 3, 4])
    assert s.epoch_len == 20


def test_get_data_simple_dataset():
    s = setting.SimpleDataset(data=np.array([0, 3, 4]), 
                              iterations=20)
    assert np.allclose(s.__getitem__(0), [0, 3, 4])


def test_epoch_len_simple_dataset():
    s = setting.SimpleDataset(data=np.array([0, 3, 4]), 
                              iterations=20)
    assert np.allclose(s.__len__(), 20)


# Test Setting
def test_create_empty_setting():
    s = setting.Setting(n_iterations=10)
    assert s.n_iterations == 10
    assert s.variables == {}
    assert isinstance(s.parameters, torch.nn.ParameterDict)
    assert s.solution_dims == {'u': 1}
    assert s.train_data == {}
    assert s.val_data == {}


def test_create_setting_with_variable():
    x = Variable(name='x', domain=Interval(0, 1))
    s = setting.Setting(variables=x)
    assert s.variables == {'x': x}


def test_create_setting_with_variable_list():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(0, 1))
    s = setting.Setting(variables=[x, t])
    assert s.variables == {'x': x, 't': t}


def test_create_setting_with_variable_dic():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(0, 1))
    s = setting.Setting(variables={'x': x, 't': t})
    assert s.variables == {'x': x, 't': t}


def test_create_setting_with_wrong_variable_type():
    with pytest.raises(TypeError):
        _ = setting.Setting(variables=2)


def test_create_setting_with_parameter():
    p = Parameter(init=1, name='p')
    s = setting.Setting(parameters=p)
    assert s.parameters['p'] == p


def test_create_setting_with_list_of_parameters():
    p = Parameter(init=1, name='p')
    D = Parameter(init=[3,0], name='D')
    s = setting.Setting(parameters=[p, D])
    assert s.parameters['p'] == p
    assert torch.equal(s.parameters['D'], D)


def test_create_setting_with_dic_of_parameters():
    p = Parameter(init=1, name='p')
    D = Parameter(init=[3,0], name='D')
    s = setting.Setting(parameters={'p': p, 'D': D})
    assert s.parameters['p'] == p
    assert torch.equal(s.parameters['D'], D)


def test_create_setting_with_wrong_parameter_type():
    with pytest.raises(TypeError):
        _ = setting.Setting(parameters=2)


def test_add_train_condition_to_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        norm=torch.nn.MSELoss())
    s = setting.Setting(variables=x)
    s.add_train_condition(c)
    assert s.train_conditions['test'] == c
    assert c.setting == s


def test_add_boundary_train_condition_to_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqBoundaryCondition(bound_condition_fun=None,
                                name='test', 
                                norm=torch.nn.MSELoss())
    s = setting.Setting(variables=x)
    s.add_train_condition(c, boundary_var='x')
    assert x.train_conditions['test'] == c
    assert x.setting == s


def test_add_val_condition_to_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        norm=torch.nn.MSELoss())
    s = setting.Setting(variables=x)
    s.add_val_condition(c)
    assert s.val_conditions['test'] == c
    assert c.setting == s


def test_add_boundary_val_condition_to_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqBoundaryCondition(bound_condition_fun=None,
                                name='test', 
                                norm=torch.nn.MSELoss())
    s = setting.Setting(variables=x)
    s.add_val_condition(c, boundary_var='x')
    assert x.val_conditions['test'] == c
    assert x.setting == s


def test_add_variable_with_condition_to_setting():
    c = DiffEqBoundaryCondition(bound_condition_fun=None,
                                name='test', 
                                norm=torch.nn.MSELoss())
    c2 = DiffEqBoundaryCondition(bound_condition_fun=None,
                                 name='test_2', 
                                 norm=torch.nn.MSELoss())
    x = Variable(name='x', domain=Interval(0, 1))
    x.add_train_condition(c)
    x.add_val_condition(c2)
    s = setting.Setting(variables=x)
    assert c2.setting == s
    assert c.setting == s


def test_get_train_conditions_of_setting():
    c = DiffEqBoundaryCondition(bound_condition_fun=None,
                                name='test', 
                                norm=torch.nn.MSELoss())
    c2 = DiffEqCondition(pde=None,
                         name='test_2', 
                         norm=torch.nn.MSELoss())
    x = Variable(name='x', domain=Interval(0, 1))
    x.add_train_condition(c)
    s = setting.Setting(variables=x,
                        train_conditions={'test_2': c2})
    out = s.get_train_conditions()
    assert isinstance(out, dict)
    assert out['x_test'] == c
    assert out['test_2'] == c2 


def test_get_val_conditions_of_setting():
    c = DiffEqBoundaryCondition(bound_condition_fun=None,
                                name='test', 
                                norm=torch.nn.MSELoss())
    c2 = DiffEqCondition(pde=None,
                         name='test_2', 
                         norm=torch.nn.MSELoss())
    x = Variable(name='x', domain=Interval(0, 1))
    x.add_val_condition(c)
    s = setting.Setting(variables=x,
                        val_conditions={'test_2': c2})
    out = s.get_val_conditions()
    assert isinstance(out, dict)
    assert out['x_test'] == c
    assert out['test_2'] == c2 


def test_setting_is_well_posed():
    s = setting.Setting()
    with pytest.raises(NotImplementedError):
        s.is_well_posed()


def test_get_input_dimension_of_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    s = setting.Setting(variables=x)
    assert s.get_dim() == 1
    s.add_variable(Variable('t', Interval(0, 2)))
    assert s.get_dim() == 2


def test_get_individual_input_dim_of_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    s = setting.Setting(variables=x)
    assert s.variable_dims == {'x': 1}
    s.add_variable(Variable('t', Interval(0, 2)))
    assert s.variable_dims == {'x': 1, 't': 1}


def test_prepare_train_data():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10,
                        norm=torch.nn.MSELoss())
    s = setting.Setting(variables=x, train_conditions={'test': c})
    assert isinstance(s.train_data['test'], dict)
    assert x.domain.is_inside(s.train_data['test']['x']).all()


def test_prepare_val_data():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-2, -1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10, 
                        norm=torch.nn.MSELoss())
    s = setting.Setting(variables=[x, t],
                        val_conditions={'test': c})
    assert isinstance(s.val_data['test'], dict)
    assert x.domain.is_inside(s.val_data['test']['x']).all()
    assert t.domain.is_inside(s.val_data['test']['t']).all()
    assert not t.domain.is_inside(s.val_data['test']['x']).any()


def test_setup_of_setting_for_np_array():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10, 
                        norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x, train_conditions={'test': c})
    s.setup()
    assert isinstance(s.train_data['test']['x'], torch.Tensor)
    assert s.train_data['test']['x'].requires_grad
    assert len(s.train_data['test']['x']) == 10


def test_setup_of_setting_without_grad_tracking():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10, 
                        track_gradients=[False],
                        norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x, val_conditions={'test': c})
    s.setup()
    assert isinstance(s.val_data['test']['x'], torch.Tensor)
    assert not s.val_data['test']['x'].requires_grad


def test_setup_of_setting_with_tensor_data():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10, 
                        track_gradients=[False],
                        norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x, val_conditions={'test': c})
    s.val_data['test']['x'] = torch.zeros(5)
    s.setup()
    assert isinstance(s.val_data['test']['x'], torch.Tensor)
    assert not s.val_data['test']['x'].requires_grad
    assert len(s.val_data['test']['x']) == 5
    

def test_setup_of_setting_with_wrong_grad_type():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        track_gradients=0,
                        norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x, val_conditions={'test': c})
    with pytest.raises(TypeError):
        s.setup()


def test_setup_of_setting_with_target_function():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        dataset_size=10,
                        norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x, val_conditions={'test': c})
    s.val_data['test']['target'] = torch.zeros(10)
    s.setup()
    assert isinstance(s.val_data['test']['x'], torch.Tensor)
    assert s.val_data['test']['x'].requires_grad
    assert len(s.val_data['test']['x']) == 10
    assert isinstance(s.val_data['test']['target'], torch.Tensor)
    assert not s.val_data['test']['target'].requires_grad
    assert len(s.val_data['test']['target']) == 10    

def test_setup_of_setting_for_validation_only():
    x = Variable(name='x', domain=Interval(0, 1))
    c = DiffEqCondition(pde=None,
                        name='test', 
                        norm=torch.nn.MSELoss())  
    c2 = DiffEqCondition(pde=None,
                         name='test_2', 
                         norm=torch.nn.MSELoss())  
    s = setting.Setting(variables=x,
                        train_conditions={'test_2': c2},
                        val_conditions={'test': c})
    s.setup('validate')
    assert isinstance(s.train_data['test_2']['x'], np.ndarray)
    assert isinstance(s.val_data['test']['x'], torch.Tensor)
    assert s.val_data['test']['x'].requires_grad


def test_get_train_dataloader_of_setting():
    s = setting.Setting()
    dataloader = s.train_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.num_workers == 0
    assert dataloader.batch_size is None 


def test_get_val_dataloader_of_setting():
    s = setting.Setting()
    dataloader = s.val_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.num_workers == 0
    assert dataloader.batch_size is None 


def test_serialize_setting():
    x = Variable(name='x', domain=Interval(0, 1))
    s = setting.Setting(variables=x)
    def f(x):
        return 0
    c = DiffEqCondition(pde=f,
                        name='test', 
                        norm=torch.nn.MSELoss())
    c2 = DiffEqCondition(pde=f,
                         name='test_2', 
                         norm=torch.nn.MSELoss())
    s.add_train_condition(c)
    s.add_val_condition(c2)
    p = Parameter(init=1, name='p')
    s.add_parameter(p)
    dct = s.serialize()
    assert dct['name'] == 'Setting'
    assert dct['n_iterations'] == 1000
    assert isinstance(dct['variables'], dict)
    assert dct['variables']['x'] == x.serialize()
    assert isinstance(dct['train_conditions'], dict)
    assert dct['train_conditions']['test'] == c.serialize()
    assert isinstance(dct['val_conditions'], dict)
    assert dct['val_conditions']['test_2'] == c2.serialize()
    assert isinstance(dct['parameters'], dict)
    assert dct['parameters']['p'] == [1]
    assert dct['solution_dims'] == {'u': 1}