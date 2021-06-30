import torch
import numpy as np
import pytest
from neural_diff_eq.problem import condition as condi
from neural_diff_eq.problem.domain.domain1D import Interval
from neural_diff_eq.problem.variables.variable import Variable

# Helper functions for testing
def model_function(input):
    return input['x']

def condition_function(model, data):
    return model - data['out']


# Test parent class
def test_create_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss(), weight=2, 
                           track_gradients=True, data_plot_variables=True)
    assert cond.name == 'test'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 2
    assert cond.track_gradients
    assert cond.data_plot_variables
    assert cond.variables is None


def test_none_methode_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    assert cond.get_data() is None
    assert cond.get_data_plot_variables() is None    


def test_new_condition_not_registered():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    assert not cond.is_registered()


def test_serialize_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    dct = cond.serialize()
    assert dct['name'] == 'test'
    assert dct['norm'] == 'MSELoss'
    assert dct['weight'] == 1


# Test DiffEqCondition
def test_create_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    assert cond.name == 'pde'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 1
    assert cond.track_gradients
    assert not cond.data_plot_variables
    assert cond.variables is None
    assert cond.dataset_size == 10000
    assert cond.pde == condition_function


def test_forward_diffeqcondition_with_MSE():
    data = {'x': torch.FloatTensor([[1, 1], [1, 0]]), 
            'out': torch.FloatTensor([[1, 1], [1, 0]])}
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    out = cond.forward(model_function, data)
    assert out == 0  
    data = {'x': torch.FloatTensor([[1, 1], [1, 0]]), 
            'out': torch.FloatTensor([[0, 1], [1, 0]])}
    out = cond.forward(model_function, data)
    assert out == 1/4  


def test_forward_diffeqcondition_with_L1Loss():
    data = {'x': torch.FloatTensor([[1, 1], [1, 0]]), 
            'out': torch.FloatTensor([[1, 1], [1, 0]])}
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.L1Loss(reduction='sum'))
    out = cond.forward(model_function, data)
    assert out == 0  
    data = {'x': torch.FloatTensor([[1, 1], [1, 0]]), 
            'out': torch.FloatTensor([[0, 1], [1, 0]])}
    out = cond.forward(model_function, data)
    assert out == 1


def test_get_data_diffeqcondition_not_registered():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(), 
                                 dataset_size=500, 
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    cond.variables = {'x': x, 't': t}
    data = cond.get_data()
    assert np.shape(data['x']) == (500, 1)
    assert np.shape(data['t']) == (500, 1)
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()


def test_serialize_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(), 
                                 dataset_size=500, 
                                 sampling_strategy='grid')
    dct = cond.serialize()
    assert dct['sampling_strategy'] == 'grid'
    assert dct['pde'] == 'condition_function'
    assert dct['dataset_size'] == 500


def test_get_data_plot_varibales_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(), 
                                 dataset_size=500, 
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    cond.variables = {'x': x, 't': t}
    assert cond.get_data_plot_variables() is None
    cond.data_plot_variables = True
    assert cond.get_data_plot_variables() == {'x': x, 't': t}
    cond.data_plot_variables = x
    assert cond.get_data_plot_variables() == x


# Test datacondition
def create_data_condition():
    return condi.DataCondition(name='test',
                               norm=torch.nn.MSELoss(), 
                               data_x={'x': torch.ones(5)}, 
                               data_u=torch.tensor([1, 2, 1, 1, 0]))
def test_create_datacondition():
    cond = create_data_condition()
    assert cond.name == 'test'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 1
    assert not cond.track_gradients
    assert not cond.data_plot_variables
    assert cond.variables is None
    assert torch.equal(cond.data_x['x'], torch.ones(5))
    assert torch.equal(cond.data_u, torch.tensor([1, 2, 1, 1, 0]))


def test_get_data_plot_varibales_datacondition():
    cond = create_data_condition()
    assert cond.get_data_plot_variables() is None


def test_serialize_datacondition():
    cond = create_data_condition()
    dct = cond.serialize()
    assert dct['name'] == 'test'
    assert dct['norm'] == 'MSELoss'
    assert dct['weight'] == 1


def test_get_data_datacondition_not_registered():
    cond = create_data_condition()
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_datacondition():
    cond = create_data_condition()
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    cond.variables = {'x': x, 't': t}
    data, target = cond.get_data()
    assert torch.equal(data['x'], torch.ones(5))
    assert torch.equal(target, torch.tensor([1, 2, 1, 1, 0]))


def test_forward_dataqcondition():
    cond = create_data_condition()
    cond.variables = {'x': 1}
    data = cond.get_data()
    out = cond.forward(model_function, data)
    assert out == 2/5  


# Test boundary conditions
def test_parent_boundary_condition():
    cond = condi.BoundaryCondition(name='test', 
                                   norm=torch.nn.MSELoss(),
                                   track_gradients=True)
    assert cond.boundary_variable is None


def test_serialize_boundary_condition():
    cond = condi.BoundaryCondition(name='test', 
                                   norm=torch.nn.MSELoss(),
                                   track_gradients=True)
    dct = cond.serialize()
    assert dct['boundary_variable'] is None


def test_get_data_plot_varibales_boundary_conditon():
    cond = condi.BoundaryCondition(name='test', 
                                   norm=torch.nn.MSELoss(),
                                   track_gradients=True)
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    cond.variables = {'x': x}
    cond.boundary_variable = t
    cond.data_plot_variables = False
    assert cond.get_data_plot_variables() is None
    cond.data_plot_variables = True
    assert cond.get_data_plot_variables() == t
    cond.data_plot_variables = x
    assert cond.get_data_plot_variables() == x


# Test dirichlet condition
def dirichlet_fun(input):
    return input['x']

def create_dirichlet():
    return condi.DirichletCondition(dirichlet_fun=dirichlet_fun,
                                    name='test diri',
                                    norm=torch.nn.MSELoss(),
                                    sampling_strategy='grid',
                                    boundary_sampling_strategy='random',
                                    weight=1.5,
                                    dataset_size=50,
                                    data_plot_variables=True)


def test_create_dirichlet_condition():
    cond = create_dirichlet()
    assert cond.dirichlet_fun == dirichlet_fun
    assert cond.name == 'test diri'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.sampling_strategy == 'grid'
    assert cond.boundary_sampling_strategy == 'random'
    assert cond.boundary_variable is None
    assert cond.weight == 1.5
    assert cond.dataset_size == 50
    assert cond.data_plot_variables 


def test_serialize_dirichlet_condition():
    cond = create_dirichlet()
    dct = cond.serialize()
    assert dct['dirichlet_fun'] == 'dirichlet_fun'
    assert dct['dataset_size'] == 50
    assert dct['sampling_strategy'] == 'grid'
    assert dct['boundary_sampling_strategy'] == 'random'


def test_get_data_dirichlet_qcondition_not_registered():
    cond = create_dirichlet()
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_dirichlet_condition():
    cond = create_dirichlet()
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    cond.variables = {'x': x, 't': t}
    cond.boundary_variable = 't'
    data, target = cond.get_data()
    assert np.shape(data['x']) == (50, 1)
    assert np.shape(data['t']) == (50, 1)
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()
    assert np.equal(data['x'], target).all()


def test_forward_dirichlet_condition():
    cond = create_dirichlet()
    data = ({'x': torch.ones((2,1))}, torch.zeros((2,1)))
    out = cond.forward(model_function, data)
    assert out.item() == 1 
    assert isinstance(out, torch.Tensor)