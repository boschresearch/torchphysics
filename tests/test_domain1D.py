import pytest
import numpy as np
from neural_diff_eq.problem.domain.domain1D import Interval

def test_interval_has_correct_bounds():
    I = Interval(1,3.5)
    assert I.low_bound == 1
    assert I.up_bound == 3.5

def test_wrong_bounds_raise_error():
    with pytest.raises(ValueError):
        I = Interval(2,0)

def test_points_inside_and_outside():
    I = Interval(0,1)
    inside = 0.5
    outside = 10
    assert I.is_inside(inside)
    assert not I.is_inside(outside)

def test_points_at_boundary():
    I = Interval(0.5,2.5)
    assert I.is_on_boundary(0.5)
    assert not I.is_on_boundary(-2)

def test_set_error_toleranz():
    I = Interval(0,1,tol=2)
    assert I.tol == 2

def test_dimension():
    I = Interval(0,1)
    assert I.dim == 1

def test_output_type():
    I = Interval(0,1)
    i_rand = I.sample_inside(1)
    b_rand = I.sample_boundary(1)
    i_grid = I.sample_inside(1, type='grid')
    b_grid = I.sample_boundary(1, type='grid')
    b_lower = I.sample_boundary(1, type='lower_bound_only')
    b_upper = I.sample_boundary(1, type='upper_bound_only')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)
    assert isinstance(b_lower[0][0], np.float32)
    assert isinstance(b_upper[0][0], np.float32)

def test_random_sampling_inside():
    np.random.seed(0)
    I = Interval(0,1)
    p = I.sample_inside(10, type='random')
    compare = [[0.5488135 ], [0.71518934], [0.60276335], [0.5448832 ],
               [0.4236548 ], [0.6458941 ], [0.4375872 ], [0.891773  ],
               [0.96366274], [0.3834415 ]]
    compare = np.array(compare).astype(np.float32)
    assert len(p) == 10
    assert all(I.is_inside(p))
    assert all(compare == p)

def test_grid_sampling_inside():
    I = Interval(0.2,5)
    p = I.sample_inside(47, type='grid')
    assert len(p) == 47
    assert all(I.is_inside(p))
    assert np.abs(p[1]-p[0]) == np.abs(p[2]-p[1])

def test_random_sampling_boundary():
    np.random.seed(0)
    I = Interval(0,6)
    p = I.sample_boundary(5, type='random')
    compare = [[0.], [6.], [6.], [0.], [6.]]
    compare = np.array(compare).astype(np.float32)
    assert len(p) == 5
    assert all(I.is_on_boundary(p))
    assert all(compare == p)

def test_grid_sampling_boundary():
    I = Interval(0,6)
    p = I.sample_boundary(200, type='grid')
    assert len(p) == 200
    assert all(I.is_on_boundary(p))
    index = np.where(np.isclose(p, 0))
    assert len(index[0]) == 100 

def test_lower_bound_sampling():
    I = Interval(-1,2)
    p = I.sample_boundary(150, type='lower_bound_only')
    assert len(p) == 150
    assert all(I.is_on_boundary(p))
    assert all(np.isclose(p,-1))

def test_upper_bound_sampling():
    I = Interval(-1,2)
    p = I.sample_boundary(150, type='upper_bound_only')
    assert len(p) == 150
    assert all(I.is_on_boundary(p))
    assert all(np.isclose(p,2))

def test_grid_for_plot():
    I = Interval(-1,2)
    p = I.grid_for_plots(150)
    assert len(p) == 150
    assert p[0] == I.low_bound
    assert p[-1] == I.up_bound
    assert all(I.is_inside(p[1:-1]))