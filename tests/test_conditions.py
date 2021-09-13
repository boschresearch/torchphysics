import torch
import numpy as np
import pytest
from torchphysics.problem import condition as condi
from torchphysics.problem import datacreator as dc
from torchphysics.problem.domain.domain1D import Interval
from torchphysics.problem.domain.domain2D import Rectangle
from torchphysics.setting import Setting
from torchphysics.problem.variables.variable import Variable

# Helper functions for testing


def model_function(input):
    return {'u': input['x']}


def condition_function(u, data):
    return u - data


# Test parent class 
def test_create_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss(), weight=2,
                           track_gradients=True, data_plot_variables=True)
    assert cond.name == 'test'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 2
    assert cond.track_gradients
    assert cond.data_plot_variables
    assert cond.setting is None


def test_none_methode_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    assert cond.get_data() is None
    assert cond.get_data_plot_variables() is None


def test_none_methode_datacreator():
    creator = dc.DataCreator(None, 1, '', {})
    assert creator.get_data() is None


def test_new_condition_not_registered():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    assert not cond.is_registered()


def test_serialize_condition():
    cond = condi.Condition(name='test', norm=torch.nn.MSELoss())
    dct = cond.serialize()
    assert dct['name'] == 'test'
    assert dct['norm'] == 'MSELoss'
    assert dct['weight'] == 1


def test_remove_nan_from_data():
    out = np.array([[np.NaN], [2]])
    batch_size = 2
    input_dic = {'x': np.array([[0], [2]]),
                 't': np.array([[1], [3]])}
    input_dic, out = condi.remove_nan(input_dic, out, batch_size)
    assert len(out) == 1
    assert out[0] == [2]
    assert input_dic['x'] == [2]
    assert input_dic['t'] == [3]


def test_remove_nan_from_data_only_for_arrays():
    out = np.array([[np.NaN], [2]])
    batch_size = 2
    input_dic = {'x': [[0], [2]],
                 't': [[1], [3]]}
    input_dic, out = condi.remove_nan(input_dic, out, batch_size)
    assert len(out) == 1
    assert out[0] == [2]
    assert np.allclose(input_dic['x'], [[0], [2]])
    assert np.allclose(input_dic['t'], [[1], [3]])


def test_get_data_len_with_int():
    assert condi.get_data_len(3) == 3


def test_get_data_len_with_list():
    assert condi.get_data_len([3, 4, 5]) == 60


def test_get_data_len_with_dic():
    assert condi.get_data_len({'x': 4, 't': 3}) == 12    


def test_get_data_len_error_for_wrong_type():
    with pytest.raises(ValueError):
        _ = condi.get_data_len('hello')


# Test DiffEqCondition
def test_create_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    assert cond.name == 'pde'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 1
    assert cond.track_gradients
    assert not cond.data_plot_variables
    assert cond.setting is None
    assert cond.datacreator_list[0].dataset_size == 10000
    assert cond.pde == condition_function


def test_add_data_points_to_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    cond.add_sample_points(sampling_strategy='grid', dataset_size=50)
    assert len(cond.datacreator_list) == 2
    assert cond.datacreator_list[1].dataset_size == 50
    assert cond.datacreator_list[1].sampling_strategy == 'grid'    


def test_forward_diffeqcondition_with_MSE():
    inp = {'x': torch.FloatTensor([[1, 1], [1, 0]]),
           'data': torch.FloatTensor([[1, 1], [1, 0]])}
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    x = Variable(name='x', domain=None)
    setting = Setting(variables={'x': x})
    cond.setting = setting
    out = cond.forward(model_function, inp)
    assert out == 0
    inp = {'x': torch.FloatTensor([[1, 1], [1, 0]]),
           'data': torch.FloatTensor([[0, 1], [1, 0]])}
    out = cond.forward(model_function, inp)
    assert out == 1/4


def test_forward_diffeqcondition_with_L1Loss():
    inp = {'x': torch.FloatTensor([[1, 1], [1, 0]]),
           'data': torch.FloatTensor([[1, 1], [1, 0]])}
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.L1Loss(reduction='sum'))
    x = Variable(name='x', domain=None)
    setting = Setting(variables={'x': x})
    cond.setting = setting
    out = cond.forward(model_function, inp)
    assert out == 0
    inp = {'x': torch.FloatTensor([[1, 1], [1, 0]]),
           'data': torch.FloatTensor([[0, 1], [1, 0]])}
    out = cond.forward(model_function, inp)
    assert out == 1


def test_get_data_diffeqcondition_not_registered():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss())
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_diffeqcondition_wrong_strategy():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 sampling_strategy='test')
    setting = Setting(variables={})
    cond.setting = setting
    with pytest.raises(NotImplementedError):
        cond.get_data()


def test_data_sampling_with_int_random_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=500,
                                 sampling_strategy='random')
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (500, 1)
    assert np.shape(data['t']) == (500, 1)
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()


def test_data_sampling_with_two_datacreators_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=500,
                                 sampling_strategy='random')
    x = Variable(name='x', domain=Interval(0, 1))
    setting = Setting(variables={'x': x})
    cond.add_sample_points(sampling_strategy='grid', dataset_size=50)
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (550, 1)
    assert x.domain.is_inside(data['x']).all()


def test_data_sampling_with_int_grid_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=100,
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (100, 1)
    assert np.shape(data['t']) == (100, 1)
    for i in range(9):
        assert data['x'][i] == data['x'][i+1]
    assert np.equal(data['t'][0:10], data['t'][10:20]).all()


def test_data_sampling_with_int_grid_divide_2D_1D_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=1000,
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (1000, 2)
    assert np.shape(data['t']) == (1000, 1)
    for i in range(9):
        assert np.equal(data['x'][i], data['x'][i+1]).all()
        assert np.equal(data['x'][100+i], data['x'][i+101]).all()
    assert np.equal(data['t'][0:100], data['t'][100:200]).all()


def test_data_sampling_with_wrong_input_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size='42',
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    with pytest.raises(TypeError):
        _ = cond.get_data()


def test_data_sampling_with_list_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=[10, 10, 5],
                                 sampling_strategy=['random', 'random', 'random'])
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    D = Variable(name='D', domain=Interval(2, 3))
    setting = Setting(variables={'x': x, 't': t, 'D': D})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (500, 1)
    assert np.shape(data['t']) == (500, 1)
    assert np.shape(data['D']) == (500, 1)
    assert t.domain.is_inside(data['t']).all()
    assert x.domain.is_inside(data['x']).all()
    assert D.domain.is_inside(data['D']).all()
    assert not x.domain.is_inside(data['t']).all()
    assert not D.domain.is_inside(data['t']).all()


def test_data_sampling_with_wrong_inputs_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=[10],
                                 sampling_strategy=34)
    x = Variable(name='x', domain=Interval(0, 1))
    setting = Setting(variables={'x': x})
    cond.setting = setting
    with pytest.raises(TypeError):
        _ = cond.get_data()


def test_data_sampling_with_dic_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size={'x': 5, 't': 10},
                                 sampling_strategy={'x': 'grid', 't': 'grid'})
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (50, 2)
    assert np.shape(data['t']) == (50, 1)
    assert t.domain.is_inside(data['t']).all()
    assert x.domain.is_inside(data['x']).all()


def test_data_sampling_with_data_fun_diffeqcondition():
    def fun(x):
        return x
    cond = condi.DiffEqCondition(pde=condition_function,
                                 data_fun=fun,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=500,
                                 sampling_strategy='random', 
                                 data_fun_whole_batch=False)
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    data = cond.get_data()
    assert np.shape(data['x']) == (500, 1)
    assert np.shape(data['t']) == (500, 1)
    assert np.shape(data['data']) == (500, 1)
    assert np.equal(data['data'], data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()


def test_serialize_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=500,
                                 sampling_strategy='grid')
    dct = cond.serialize()
    assert dct['sampling_strategy'] == ['grid']
    assert dct['pde'] == 'condition_function'
    assert dct['dataset_size'] == [500]


def test_get_data_plot_varibales_diffeqcondition():
    cond = condi.DiffEqCondition(pde=condition_function,
                                 norm=torch.nn.MSELoss(),
                                 dataset_size=500,
                                 sampling_strategy='grid')
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, 1))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    assert cond.get_data_plot_variables() is None
    cond.data_plot_variables = True
    assert cond.get_data_plot_variables() == {'x': x, 't': t}
    cond.data_plot_variables = x
    assert cond.get_data_plot_variables() == x


# Test datacondition
def create_data_condition():
    return condi.DataCondition(name='test',
                               norm=torch.nn.MSELoss(),
                               data_inp={'x': torch.ones(5)},
                               data_out=torch.tensor([1, 2, 1, 1, 0]))


def test_create_datacondition():
    cond = create_data_condition()
    assert cond.name == 'test'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.weight == 1
    assert not cond.track_gradients
    assert not cond.data_plot_variables
    assert cond.setting is None
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
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    inp = cond.get_data()
    assert torch.equal(inp['x'], torch.ones(5))
    assert torch.equal(inp['target'], torch.tensor([1, 2, 1, 1, 0]))


def test_forward_datacondition():
    cond = create_data_condition()
    x = Variable(name='x', domain=None)
    setting = Setting(variables={'x': x})
    cond.setting = setting
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
def dirichlet_fun(**input):
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
    assert cond.datacreator_list[0].sampling_strategy == 'grid'
    assert cond.datacreator_list[0].boundary_sampling_strategy == 'random'
    assert cond.boundary_variable is None
    assert cond.weight == 1.5
    assert cond.datacreator_list[0].dataset_size == 50
    assert cond.data_plot_variables


def test_serialize_dirichlet_condition():
    cond = create_dirichlet()
    dct = cond.serialize()
    assert dct['dirichlet_fun'] == 'dirichlet_fun'
    assert dct['dataset_size'] == [50]
    assert dct['sampling_strategy'] == ['grid']
    assert dct['boundary_sampling_strategy'] == ['random']


def test_get_data_dirichlet_condition_not_registered():
    cond = create_dirichlet()
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_dirichlet_condition():
    cond = create_dirichlet()
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-3, -2))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    cond.boundary_variable = 't'
    data  = cond.get_data()
    assert np.shape(data['x']) == (64, 1)
    assert np.shape(data['t']) == (64, 1)
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()
    assert np.equal(data['x'], data['target']).all()


def test_forward_dirichlet_condition():
    x = Variable(name='x', domain=Interval(0, 1))
    setting = Setting(variables={'x': x})
    cond = create_dirichlet()
    cond.setting = setting
    data = ({'x': torch.ones((2, 1)), 'target': torch.zeros((2, 1))})
    out = cond.forward(model_function, data)
    assert out.item() == 1
    assert isinstance(out, torch.Tensor)


def test_boundary_data_creation_random_random_int():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-3, -2))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=10,
                                     sampling_strategy='random',
                                     boundary_sampling_strategy='random')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (10, 1)
    assert np.shape(data['t']) == (10, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_grid_random_int():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-3, -2))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=25,
                                     sampling_strategy='grid',
                                     boundary_sampling_strategy='random')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (25, 1)
    assert np.shape(data['t']) == (25, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -3 or data['t'][i] == -2
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_random_grid_int():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=25,
                                     sampling_strategy='random',
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (26, 1)
    assert np.shape(data['t']) == (26, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -1 or data['t'][i] == -0.1
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_grid_grid_int():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=30,
                                     sampling_strategy='grid',
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (30, 1)
    assert np.shape(data['t']) == (30, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -1 or data['t'][i] == -0.1
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_random_lower_bound_int():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=30,
                                     sampling_strategy='random',
                                     boundary_sampling_strategy='lower_bound_only')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (30, 1)
    assert np.shape(data['t']) == (30, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -1
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_with_list():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=[30, 2],
                                     sampling_strategy=['grid'],
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (60, 1)
    assert np.shape(data['t']) == (60, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -1 or data['t'][i] == -0.1
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_with_dic():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size={'x': 10, 't': 1},
                                     sampling_strategy={'x': 'grid'},
                                     boundary_sampling_strategy='lower_bound_only')
    creator.boundary_variable = 't'
    data = creator.get_data()
    assert np.shape(data['x']) == (10, 1)
    assert np.shape(data['t']) == (10, 1)
    assert x.domain.is_inside(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['t'])):
        assert data['t'][i] == -1
    assert not x.domain.is_inside(data['t']).all()


def test_boundary_data_creation_with_wrong_type():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size={'x': 10, 't': 1},
                                     sampling_strategy=23,
                                     boundary_sampling_strategy='lower_bound_only')
    with pytest.raises(TypeError):
        _ = creator.get_data()


def test_boundary_data_creation_with_3_inputs():
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    D = Variable(name='D', domain=Interval(3, 4))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t, 'D': D},
                                     dataset_size=[10, 1, 10],
                                     sampling_strategy='grid',
                                     boundary_sampling_strategy='lower_bound_only')
    creator.boundary_variable = 'x'
    data = creator.get_data()
    assert np.shape(data['x']) == (100, 1)
    assert np.shape(data['D']) == (100, 1)
    assert np.shape(data['t']) == (100, 1)
    assert x.domain.is_inside(data['x']).all()
    assert D.domain.is_inside(data['D']).all()
    assert t.domain.is_inside(data['t']).all()
    for i in range(len(data['x'])):
        assert data['x'][i] == 0
    assert not x.domain.is_inside(data['t']).all()
    assert not x.domain.is_inside(data['D']).all()


def test_boundary_data_creation_with_2D_boundary_grid_grid():
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    D = Variable(name='D', domain=Interval(3, 4))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t, 'D': D},
                                     dataset_size=[10, 10, 10],
                                     sampling_strategy='grid',
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 'x'
    data = creator.get_data()
    assert np.shape(data['x']) == (1000, 2)
    assert np.shape(data['D']) == (1000, 1)
    assert np.shape(data['t']) == (1000, 1)
    assert x.domain.is_on_boundary(data['x']).all()
    assert D.domain.is_inside(data['D']).all()
    assert t.domain.is_inside(data['t']).all()
    assert not D.domain.is_inside(data['t']).all()


def test_boundary_data_creation_with_2D_boundary_grid_grid_int():
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=1000,
                                     sampling_strategy={'t': 'grid', 'x': 'grid'},
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 'x'
    data = creator.get_data()
    assert np.shape(data['x']) == (1000, 2)
    assert np.shape(data['t']) == (1000, 1)
    assert x.domain.is_on_boundary(data['x']).all()
    assert t.domain.is_inside(data['t']).all()
    assert np.equal(data['t'][0:10], data['t'][10:20]).all()
    for i in range(9):
        assert np.equal(data['x'][i], data['x'][i+1]).all()


def test_boundary_data_creation_with_2D_boundary_random_grid_int():
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=100,
                                     sampling_strategy='random',
                                     boundary_sampling_strategy='grid')
    creator.boundary_variable = 'x'
    data = creator.get_data()
    assert np.shape(data['x']) == (125, 2)
    assert np.shape(data['t']) == (125, 1)
    assert x.domain.is_on_boundary(data['x']).all()
    assert t.domain.is_inside(data['t']).all()


def test_boundary_data_creation_with_2D_boundary_grid_random_int():
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-1, -0.1))
    creator = dc.BoundaryDataCreator(variables={'x': x, 't': t},
                                     dataset_size=100,
                                     sampling_strategy='grid',
                                     boundary_sampling_strategy='random')
    creator.boundary_variable = 'x'
    data = creator.get_data()
    assert np.shape(data['x']) == (125, 2)
    assert np.shape(data['t']) == (125, 1)
    assert x.domain.is_on_boundary(data['x']).all()
    assert t.domain.is_inside(data['t']).all()


# Test neumann conditions
def neumann_fun(**input):
    return np.zeros_like(input['t'])


def create_neumann():
    return condi.NeumannCondition(neumann_fun=neumann_fun,
                                  name='test neumann',
                                  norm=torch.nn.MSELoss(),
                                  sampling_strategy='grid',
                                  boundary_sampling_strategy='grid',
                                  weight=1,
                                  dataset_size=50,
                                  data_plot_variables=True)


def test_create_neumann_condition():
    cond = create_neumann()
    assert cond.neumann_fun == neumann_fun
    assert cond.name == 'test neumann'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.datacreator_list[0].sampling_strategy == 'grid'
    assert cond.datacreator_list[0].boundary_sampling_strategy == 'grid'
    assert cond.boundary_variable is None
    assert cond.weight == 1
    assert cond.datacreator_list[0].dataset_size == 50
    assert cond.data_plot_variables


def test_serialize_neumann_condition():
    cond = create_neumann()
    dct = cond.serialize()
    assert dct['neumann_fun'] == 'neumann_fun'
    assert dct['dataset_size'] == [50]
    assert dct['sampling_strategy'] == ['grid']
    assert dct['boundary_sampling_strategy'] == ['grid']


def test_get_data_neumann_condition_not_registered():
    cond = create_neumann()
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_neumann_condition():
    cond = create_neumann()
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-3, -2))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    cond.boundary_variable = 'x'
    inp = cond.get_data()
    assert np.shape(inp['x']) == (64, 2)
    assert np.shape(inp['t']) == (64, 1)
    assert np.shape(inp['normal']) == (64, 2)
    assert np.shape(inp['target']) == (64, 1)
    assert t.domain.is_inside(inp['t']).all()
    assert x.domain.is_on_boundary(inp['x']).all()
    for i in range(len(inp['normal'])):
        assert np.isclose(np.linalg.norm(inp['normal'][i]), 1)
        assert np.isclose(inp['target'][i], 0)
    for i in range(len(inp['normal'])):
        new_normal = x.domain.boundary_normal([inp['x'][i]])
        assert np.allclose(new_normal, inp['normal'][i])


def test_forward_neumann_condition():
    cond = create_neumann()
    x = Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    t = Variable(name='t', domain=Interval(-3, -2))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    cond.boundary_variable = 'x'
    inp = cond.get_data()
    target = torch.from_numpy(inp['target'])
    normals = torch.from_numpy(inp['normal'])
    inp['x'] = torch.from_numpy(inp['x'])
    inp['x'].requires_grad = True
    data = {**inp, 'target': target, 'normal': normals}
    out = cond.forward(model_function, data)
    assert out.item() == 1
    assert isinstance(out, torch.Tensor)
    norm = torch.nn.MSELoss()
    assert torch.isclose(out, norm(normals.sum(dim=1, keepdim=True), target))


# Test DiffEqBoundaryCondition for arbitrary boundary conditions
def create_arbitrary():
    return condi.DiffEqBoundaryCondition(bound_condition_fun=dirichlet_fun,
                                         name='test arbitrary',
                                         norm=torch.nn.MSELoss(),
                                         sampling_strategy='random',
                                         boundary_sampling_strategy='lower_bound_only',
                                         dataset_size=5)


def test_create_diffEqBoundary_condition():
    cond = create_arbitrary()
    assert cond.bound_condition_fun == dirichlet_fun
    assert cond.name == 'test arbitrary'
    assert isinstance(cond.norm, torch.nn.MSELoss)
    assert cond.datacreator_list[0].sampling_strategy == 'random'
    assert cond.datacreator_list[0].boundary_sampling_strategy == 'lower_bound_only'
    assert cond.boundary_variable is None
    assert cond.weight == 1
    assert cond.datacreator_list[0].dataset_size == 5
    assert cond.data_plot_variables
    assert cond.data_fun is None


def test_serialize_diffEqBoundary_condition():
    cond = create_arbitrary()
    dct = cond.serialize()
    assert dct['bound_condition_fun'] == 'dirichlet_fun'
    assert dct['dataset_size'] == [5]
    assert dct['sampling_strategy'] == ['random']
    assert dct['boundary_sampling_strategy'] == ['lower_bound_only']
    cond.data_fun = dirichlet_fun
    dct = cond.serialize()
    assert dct['data_fun'] == 'dirichlet_fun'


def test_get_data_diffEqBoundary_condition_not_registered():
    cond = create_arbitrary()
    with pytest.raises(RuntimeError):
        cond.get_data()


def test_get_data_diffEqBoundary_condition():
    cond = create_arbitrary()
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-3, -2))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    cond.boundary_variable = 't'
    inp = cond.get_data()
    assert np.shape(inp['x']) == (5, 1)
    assert np.shape(inp['t']) == (5, 1)
    assert np.equal(inp['normal'], [[-1], [-1], [-1], [-1], [-1]]).all()
    assert not x.domain.is_inside(inp['t']).all()


def test_get_data_with_target_diffEqBoundary_condition():
    cond = create_arbitrary()
    x = Variable(name='x', domain=Interval(0, 1))
    t = Variable(name='t', domain=Interval(-3, -2))
    setting = Setting(variables={'x': x, 't': t})
    cond.setting = setting
    cond.boundary_variable = 't'
    cond.data_fun = dirichlet_fun
    inp = cond.get_data()
    assert np.shape(inp['x']) == (5, 1)
    assert np.shape(inp['data']) == (5, 1)
    assert np.shape(inp['t']) == (5, 1)
    assert np.equal(inp['normal'], [[-1], [-1], [-1], [-1], [-1]]).all()
    assert not x.domain.is_inside(inp['t']).all()
    assert np.equal(inp['x'], inp['data']).all()


def test_forward_diffEqBoundary_condition_with_MSE():
    def condition_function(u, data, normal):
        return u - data
    data = {'x': torch.FloatTensor([[1, 1], [1, 0]]),
            'data': torch.FloatTensor([[1, 1], [1, 0]])}
    normals = [[1, 0], [1, 0]]
    data_comb = {**data, 'normal': normals}
    cond = create_arbitrary()
    x = Variable(name='x', domain=None)
    setting = Setting(variables={'x': x})
    cond.setting = setting
    cond.bound_condition_fun = condition_function
    out = cond.forward(model_function, data_comb)
    assert out == 0
    cond.data_fun = 2  # data_fun not None
    target = torch.FloatTensor([[2, 0], [1, 0]])

    def condition_function(u, data, normal):
        return u - target

    data_comb = {**data, 'data': target, 'normal': normals}
    cond.bound_condition_fun = condition_function
    out = cond.forward(model_function, data_comb)
    assert out == 1/2
