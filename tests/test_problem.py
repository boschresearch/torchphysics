import pytest
from torchphysics.problem import Variable
from torchphysics.problem import problem
from torchphysics.problem.condition import BoundaryCondition, Condition
from torchphysics.setting import Setting
from torchphysics.problem.domain.domain import Domain


# Test Problem
def test_create_empty_problem():
    prob = problem.Problem(train_conditions={}, val_conditions={})
    assert prob.train_conditions == {}
    assert prob.val_conditions == {}


def test_create_problem_wrong_conditions():
    with pytest.raises(TypeError):
        _ = problem.Problem(train_conditions=3, val_conditions={})


def test_none_methods_problem():
    prob = problem.Problem(train_conditions={}, val_conditions={})
    assert prob.serialize() is None
    assert prob.get_dim() is None
    assert prob.get_train_conditions() is None
    assert prob.get_val_conditions() is None
    assert prob.add_train_condition(1) is None
    assert prob.add_val_condition(1) is None
    assert prob.is_well_posed() is None


# Test Variable:
def test_create_variable():
    vari = Variable(name='test', domain=None)
    assert vari.name == 'test'
    assert vari.domain is None
    assert vari.setting is None
    assert vari.train_conditions == {}
    assert vari.val_conditions == {}
    assert vari.order == 0


def test_create_variable_with_conditions():
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    vari = Variable(name='test', domain=None, 
                    train_conditions=condi, val_conditions={})
    assert vari.train_conditions['test cond'] == condi


def test_create_variable_with_list_of_conditions():
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    condi_2 = BoundaryCondition(name='test cond 2', norm=None,
                                track_gradients=True)
    vari = Variable(name='test', domain=None, 
                    train_conditions=[condi, condi_2])
    assert vari.train_conditions['test cond'] == condi
    assert vari.train_conditions['test cond 2'] == condi_2


def test_create_variable_with_dic_of_conditions():
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    condi_2 = BoundaryCondition(name='test cond 2', norm=None,
                                track_gradients=True)
    vari = Variable(name='test', domain=None, 
                    train_conditions={'1': condi, '2': condi_2})
    assert vari.train_conditions['test cond'] == condi
    assert vari.train_conditions['test cond 2'] == condi_2


def test_add_wrong_train_condition_variable():
    vari = Variable(name='test', domain=None)
    with pytest.raises(AssertionError):
        vari.add_train_condition(Condition(name='test', norm=None))


def test_cant_add_train_condition_two_times_to_variable():
    vari = Variable(name='test', domain=None)
    vari.add_train_condition(BoundaryCondition(name='test', norm=None,
                                               track_gradients=True))
    with pytest.raises(AssertionError):
        vari.add_train_condition(BoundaryCondition(name='test', norm=None,
                                                   track_gradients=True))


def test_add_wrong_validation_condition_variable():
    vari = Variable(name='test', domain=None)
    with pytest.raises(AssertionError):
        vari.add_val_condition(Condition(name='test', norm=None))


def test_cant_add_valiation_condition_two_times_to_variable():
    vari = Variable(name='test', domain=None)
    vari.add_val_condition(BoundaryCondition(name='test', norm=None,
                                             track_gradients=True))
    with pytest.raises(AssertionError):
        vari.add_val_condition(BoundaryCondition(name='test', norm=None,
                                                 track_gradients=True))


def test_add_train_condition_in_variable():
    vari = Variable(name='test', domain=None)
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    vari.add_train_condition(condi)
    assert condi.boundary_variable == 'test'
    assert vari.train_conditions['test cond'] == condi
    assert vari.val_conditions == {}


def test_add_validation_condition_in_variable():
    vari = Variable(name='test', domain=None)
    condi = BoundaryCondition(name='test cond', norm=None,
                            track_gradients=True)
    vari.add_val_condition(condi)
    assert condi.boundary_variable == 'test'
    assert vari.val_conditions['test cond'] == condi
    assert vari.train_conditions == {}


def test_get_train_conditions():
    vari = Variable(name='test', domain=None)
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    vari.add_train_condition(condi)
    condi_2 = BoundaryCondition(name='test cond 2', norm=None,
                                track_gradients=True)
    vari.add_train_condition(condi_2)
    dic = vari.get_train_conditions()
    assert dic['test cond'] == condi
    assert dic['test cond 2'] == condi_2


def test_get_validation_conditions():
    vari = Variable(name='test', domain=None)
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    vari.add_val_condition(condi)
    condi_2 = BoundaryCondition(name='test cond 2', norm=None,
                                track_gradients=True)
    vari.add_val_condition(condi_2)
    dic = vari.get_val_conditions()
    assert dic['test cond'] == condi
    assert dic['test cond 2'] == condi_2


def test_well_posed_variable():
    vari = Variable(name='test', domain=None)
    with pytest.raises(NotImplementedError):
        vari.is_well_posed() # Add a test if implemented!


def test_get_dim_of_variable():
    vari = Variable(name='test', domain=Domain(dim=4, volume=2, surface=1, tol=0))
    assert vari.get_dim() == 4


def test_serialize_variable():
    condi = BoundaryCondition(name='test cond', norm=None,
                              track_gradients=True)
    d = Domain(dim=4, volume=2, surface=1, tol=0)
    vari = Variable(name='test', domain=d, 
                    train_conditions={'test cond': condi}, 
                    val_conditions={'test cond': condi})
    dct = vari.serialize()
    assert dct['name'] == 'test'
    assert dct['domain'] == d.serialize()
    assert dct['train_conditions']['test cond'] == condi.serialize()
    assert dct['val_conditions']['test cond'] == condi.serialize()


# Test Setting
def test_empty_init():
    x = Variable(name='x',
                 domain=None)
    c = BoundaryCondition(name='c',
                          norm=None,
                          track_gradients=False)
    x.add_train_condition(c)
    setup = Setting()
    setup.add_variable(x)
    assert setup.variables['x'].setting.variables == {'x': x}
    assert c.setting.variables == {'x': x}

    d = BoundaryCondition(name='d',
                          norm=None,
                          track_gradients=False)
    y = Variable(name='y',
                 domain=None,
                 train_conditions=d)
    setup.add_variable(y)
    assert x.setting.variables == {'x': x, 'y': y}
    assert y.setting.variables == {'x': x, 'y': y}
    assert c.setting.variables == {'x': x, 'y': y}
    assert c.boundary_variable == x.name
    assert d.boundary_variable == y.name
    assert d.setting.variables == {'x': x, 'y': y}
    # test correct output of get_conditions
    assert setup.get_train_conditions() == {'x_c': c, 'y_d': d}


def test_full_init():
    c = BoundaryCondition(name='c',
                          norm=None,
                          track_gradients=False)
    x = Variable(name='x',
                 domain=None,
                 val_conditions=c)
    d = BoundaryCondition(name='d',
                          norm=None,
                          track_gradients=False)
    y = Variable(name='y',
                 domain=None)
    y.add_val_condition(d)

    e = Condition(name='e',
                  norm=None,
                  track_gradients=False)
    setup = Setting((x, y), val_conditions=e)

    assert setup.variables['x'].setting.variables == {'x': x, 'y': y}
    assert setup.get_val_conditions() == {'e': e, 'x_c': c, 'y_d': d}
    assert c.setting.variables == {'x': x, 'y': y}
    assert d.setting.variables == {'x': x, 'y': y}
    assert e.setting.variables == {'x': x, 'y': y}

    f = BoundaryCondition(name='f',
                          norm=None,
                          track_gradients=False)

    setup.add_val_condition(f, boundary_var='x')

    assert f.setting.variables == {'x': x, 'y': y}
    assert setup.get_val_conditions() == {'e': e, 'x_c': c, 'y_d': d, 'x_f': f}
    assert d.setting.variables == {'x': x, 'y': y}
    assert y.setting.variables == {'x': x, 'y': y}
