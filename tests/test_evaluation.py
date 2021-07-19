import torch
import numpy as np
from torchphysics.problem.variables import variable
from torchphysics.problem.domain.domain1D import Interval
from torchphysics.problem.domain.domain2D import Rectangle
import torchphysics.utils.evaluation as eval


def helper_function(input):
    return torch.exp(input['t'])*torch.sin(np.pi*input['x'])


def test_max_min_inside():
    x = variable.Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    mini, maxi = eval.get_min_max_inside(model=helper_function, domain_variable=x, 
                                         resolution=2500, 
                                         dic_for_other_variables={'t': 0})
    assert np.isclose(maxi, 1, atol=1e-03)
    assert np.isclose(mini, 0.061, atol=1e-02)
    mini, maxi = eval.get_min_max_inside(model=helper_function, domain_variable=x, 
                                         resolution=2500, 
                                         dic_for_other_variables={'t': 1})
    assert np.isclose(maxi, np.exp(1), atol=1e-02)
    assert np.isclose(mini, np.exp(1)*0.061, atol=1e-02)


def test_max_min_inside_1D():
    x = variable.Variable(name='x', domain=Interval(0, 2))
    mini, maxi = eval.get_min_max_inside(model=helper_function, domain_variable=x, 
                                         resolution=2500, 
                                         dic_for_other_variables={'t': 0})
    assert np.isclose(maxi, 1, atol=1e-03)
    assert np.isclose(mini, -1, atol=1e-03)
    def helper_function2(input):
        return input['x']
    mini, maxi = eval.get_min_max_inside(model=helper_function2, domain_variable=x, 
                                         resolution=2500)
    assert np.isclose(maxi, 2, atol=1e-03)
    assert np.isclose(mini, 0, atol=1e-03)


def test_max_min_boundary():
    x = variable.Variable(name='x', domain=Rectangle([0, 0], [1, 0], [0, 1]))
    mini, maxi = eval.get_min_max_boundary(model=helper_function, boundary_variable=x, 
                                           resolution=2500, 
                                           dic_for_other_variables={'t': 0})
    assert np.isclose(maxi, 1)
    assert np.isclose(mini, 0, atol=1e-07)

def test_max_min_boundary_1D():
    x = variable.Variable(name='x', domain=Interval(0, 1/2))
    mini, maxi = eval.get_min_max_boundary(model=helper_function, boundary_variable=x, 
                                           resolution=2500, 
                                           dic_for_other_variables={'t': 0})
    assert np.isclose(maxi, 1)
    assert np.isclose(mini, 0, atol=1e-07)
    mini, maxi = eval.get_min_max_boundary(model=helper_function, boundary_variable=x, 
                                           resolution=2500, 
                                           dic_for_other_variables={'t': 2})
    assert np.isclose(maxi, np.exp(2), atol=1e-03)
    assert np.isclose(mini, 0, atol=1e-07)