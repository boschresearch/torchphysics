from neural_diff_eq.problem import Setting, Variable
from neural_diff_eq.problem.condition import BoundaryCondition, Condition


def test_empty_init():
    x = Variable(name='x',
                 domain=None)
    c = BoundaryCondition(name='c',
                          norm=None,
                          track_gradients=False)
    x.add_train_condition(c)
    setup = Setting()
    setup.add_variable(x)
    assert setup.variables['x'].context == {'x': x}
    assert c.variables == {'x': x}

    d = BoundaryCondition(name='d',
                          norm=None,
                          track_gradients=False)
    y = Variable(name='y',
                 domain=None,
                 train_conditions=d)
    setup.add_variable(y)
    assert x.context == {'x': x, 'y': y}
    assert y.context == {'x': x, 'y': y}
    assert c.variables == {'x': x, 'y': y}
    assert c.boundary_variable == x.name
    assert d.boundary_variable == y.name
    assert d.variables == {'x': x, 'y': y}
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

    assert setup.variables['x'].context == {'x': x, 'y': y}
    assert setup.get_val_conditions() == {'e': e, 'x_c': c, 'y_d': d}
    assert c.variables == {'x': x, 'y': y}
    assert d.variables == {'x': x, 'y': y}
    assert e.variables == {'x': x, 'y': y}

    f = BoundaryCondition(name='f',
                          norm=None,
                          track_gradients=False)

    setup.add_val_condition(f, boundary_var='x')

    assert f.variables == {'x': x, 'y': y}
    assert setup.get_val_conditions() == {'e': e, 'x_c': c, 'y_d': d, 'x_f': f}
    assert d.variables == {'x': x, 'y': y}
    assert y.context == {'x': x, 'y': y}
