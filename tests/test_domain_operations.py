import pytest
import numpy as np

from torchphysics.problem.domain.domain1D import Interval
from torchphysics.problem.domain.domain2D import (Rectangle, Circle,
                                                  Triangle, Polygon2D)
from torchphysics.problem.domain.domain_operations import (Cut, Union,
                                                             Intersection,
                                                             Domain_operation)


def _create_domains():
    R = Rectangle([0, 0], [0, 1], [1, 0])
    C = Circle([0, 0], 0.5)
    return R, C


def test_none_by_domain_operation():
    DO = Domain_operation(dim=0, volume=0, surface=0, tol=0)
    assert DO._approximate_volume(2) is None 
    assert DO._approximate_surface(2) is None 


def test_outline_domain_operation_for_not_implemented():
    DO = Domain_operation(dim=0, volume=0, surface=0, tol=0)
    with pytest.raises(NotImplementedError):
        _ = DO.construct_shapely(None)

# Test Cut
def test_cut():
    R, C = _create_domains()
    Cut(R, C)


def test_base_domain_and_cut_domain():
    R, C = _create_domains()
    cut = Cut(R, C)
    assert cut.base == R
    assert cut.cut == C   


def test_cant_cut_intervals():
    I1 = Interval(0,1)
    I2 = Interval(0,0.5)
    with pytest.raises(ValueError):
        _ = Cut(I1, I2)

def test_cant_cut_different_dim():
    I1 = Interval(0,1)
    R = Rectangle([0, 0], [0, 1], [1, 0])
    with pytest.raises(ValueError):
        _ = Cut(R, I1)


def test_cut_new_dim():
    R, C = _create_domains()
    cut = Cut(R, C)
    assert cut.dim == 2


def test_cut_tol():
    R, C = _create_domains()
    cut = Cut(R, C)
    assert cut.tol == R.tol


def test_cut_surface_approxi():
    R, C = _create_domains()
    cut = Cut(R, C)
    assert np.isclose(cut.surface, 3+np.pi/4, atol=1e-01)
    cut = Cut(R, C, n=10000)
    assert np.isclose(cut.surface, 3+np.pi/4, atol=1e-03)
    cut = Cut(C, R)
    assert np.isclose(cut.surface, 1+3*np.pi/4, atol=1e-01)


def test_cut_volume_approxi():
    R, C = _create_domains()
    cut = Cut(R, C)
    assert np.isclose(cut.volume, 1-np.pi/4*0.5**2, atol=1e-01)
    cut = Cut(C, R)
    assert np.isclose(cut.volume, 3*np.pi/4*0.5**2, atol=1e-01)


def test_cut_points_inside_and_outside():
    R, C = _create_domains()
    cut = Cut(R, C)
    points = [[0, 0], [0, 0.5], [0.5, 0.5], [1, 1.1], [0.25, 0.7]]
    inside = cut.is_inside(points)
    assert inside[2]
    assert inside[4]
    assert not any(inside[0:2])
    assert not inside[3]
    points = C.sample_inside(500)
    inside = cut.is_inside(points)
    assert not any(inside)


def test_cut_on_boundary():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = [[0.5, 0.5], [0.25, 0.7], [0, 0], [1, 1], [0, 0.51]]
    on_bound = cut.is_on_boundary(points)
    assert not any(on_bound[0:3])
    assert all(on_bound[3:5])


def test_random_sampling_inside_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_inside(421)
    assert np.shape(points) == (421, 2)
    assert cut.is_inside(points).all()
    assert not C.is_inside(points).any()
    assert R.is_inside(points).all()
    cut = Cut(C, R)    
    points = cut.sample_inside(421)
    assert np.shape(points) == (421, 2)
    assert cut.is_inside(points).all()
    assert not R.is_inside(points).any()
    assert C.is_inside(points).all()


def test_grid_sampling_inside_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_inside(423, type='grid')
    assert np.shape(points) == (423, 2)
    assert cut.is_inside(points).all()
    assert not C.is_inside(points).any()
    assert R.is_inside(points).all()
    cut = Cut(C, R)    
    points = cut.sample_inside(21, type='grid')
    assert np.shape(points) == (21, 2)
    assert cut.is_inside(points).all()
    assert not R.is_inside(points).any()
    assert C.is_inside(points).all()


def test_lhs_sampling_inside_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_inside(40, type='lhs')
    assert np.shape(points) == (40, 2)
    assert cut.is_inside(points).all()
    assert isinstance(points[0][0], np.float32)


def test_grid_sampling_inside_other_cut():
    R, C = _create_domains()
    cut = Cut(C, R)    
    cut.volume /= 4
    points = cut.sample_inside(21, type='grid')
    assert np.shape(points) == (21, 2)
    assert cut.is_inside(points).all()
    assert not R.is_inside(points).any()
    assert C.is_inside(points).all()


def test_random_sampling_boundary_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_boundary(10)
    assert np.shape(points) == (10, 2)
    assert cut.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all()
    cut = Cut(C, R)    
    points = cut.sample_boundary(121)
    assert np.shape(points) == (121, 2)
    assert cut.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all() 


def test_grid_sampling_boundary_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_boundary(10, type='grid')
    assert np.shape(points) == (10, 2)
    assert cut.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all()
    cut = Cut(C, R)    
    points = cut.sample_boundary(121, type='grid')
    assert np.shape(points) == (121, 2)
    assert cut.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all()


def test_normal_sampling_boundary_cut():
    R, C = _create_domains()
    cut = Cut(R, C)    
    points = cut.sample_boundary(10, type='normal', sample_params={'mean': [1, 1], 
                                                                   'cov': 0.1})
    assert all(cut.is_on_boundary(points))
    assert np.shape(points) == (10, 2)
    assert isinstance(points[0][0], np.float32)   


def test_grid_for_plots_cut():
    R, C = _create_domains()
    cut = Cut(R, C)
    points = cut.grid_for_plots(200)
    inside = cut.is_inside(points)
    on_bound = cut.is_on_boundary(points)
    assert np.logical_or(inside, on_bound).all() 


def test_empty_cut():
    R = _create_domains()[0]
    T = Triangle([-1, -1], [-0.5, -0.5], [-0.2, 0.2])
    cut = Cut(T, R)
    points = cut.grid_for_plots(100)
    assert all(T.is_inside(points))
    points = cut.sample_inside(50, type='grid')
    assert all(T.is_inside(points))
    assert T.volume == cut.volume
    assert T.surface == cut.surface


def test_normals_of_cut_1():
    R, C = _create_domains()
    cut = Cut(R, C)
    points = [[1, 0.5], [0.3, 1], [0, 0.7], 
              [0.5*np.cos(np.pi/4), 0.5*np.sin(np.pi/4)]]  
    normals = cut.boundary_normal(points)
    assert np.allclose(normals[0], [1, 0]) 
    assert np.allclose(normals[1], [0, 1]) 
    assert np.allclose(normals[2], [-1, 0]) 
    assert np.allclose(normals[3], [-np.cos(np.pi/4), -np.sin(np.pi/4)]) 
    point = [[0.6, 0]]
    normal = cut.boundary_normal(point)
    assert np.allclose(normal[0], [0, -1]) 
    point = [[0.5*np.cos(np.pi/5), 0.5*np.sin(np.pi/5)]]
    normal = cut.boundary_normal(point)
    assert np.allclose(normal[0], [-np.cos(np.pi/5), -np.sin(np.pi/5)]) 


def test_normals_of_cut_2():
    R, C = _create_domains()
    cut = Cut(C, R)
    points = [[0, 0], [0, 0.2], [0.2, 0], 
              [-0.5*np.cos(np.pi/4), -0.5*np.sin(np.pi/4)]]  
    normals = cut.boundary_normal(points)
    assert np.allclose(normals[0], [1/np.sqrt(2), 1/np.sqrt(2)])  
    assert np.allclose(normals[1], [1, 0]) 
    assert np.allclose(normals[2], [0, 1]) 
    assert np.allclose(normals[3], [-np.cos(np.pi/4), -np.sin(np.pi/4)]) 


def test_cut_hole():
    R = _create_domains()[0]
    C = Circle([0.5, 0.5], 0.3)
    cut = Cut(R, C)
    b = R.sample_boundary(50, type='grid')
    assert cut.is_on_boundary(b).all()
    b = C.sample_boundary(50, type='grid')
    assert cut.is_on_boundary(b).all()
    i = C.sample_inside(50)
    assert not cut.is_inside(i).any()


def test_cut_many_times():
    R, C = _create_domains()
    C = Cut(R, C)
    C = Cut(C, Circle([1, 1], 0.5))
    C = Cut(C, Circle([0, 1], 0.5))
    C = Cut(C, Circle([0.5, 0.5], 0.05))
    b = C.sample_inside(20)
    i = C.sample_boundary(20)
    assert C.is_inside(b).all()
    assert C.is_on_boundary(i).all()


def test_cut_bounds():
    R, C = _create_domains()
    C = Cut(R, C)
    bounds = C._compute_bounds()
    assert np.allclose(bounds, [0, 1, 0, 1])


def test_outline_cut():
    R = Rectangle([0, 0], [2, 0], [0, 2])
    R2 = Rectangle([0.5, 0.5], [1, 0.5], [0.5, 1])
    C = Cut(R, R2)
    outline = C.outline()
    assert isinstance(outline, list)
    assert np.allclose(outline[0], [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]])
    assert np.allclose(outline[1], [[0.5, 0.5], [1, 0.5], [1, 1],
                                    [0.5, 1], [0.5, 0.5]])


def test_serialize_cut():
    R, C = _create_domains()
    C = Cut(R, C)
    dct = C.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == '(Rectangle - Circle)'
    C = Cut(C, Circle([1, 1], 0.5))
    dct = C.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == '((Rectangle - Circle) - Circle)'


# Test Union
def test_union():
    R, C = _create_domains()
    Union(R, C)


def test_domain_order_union():
    R, C = _create_domains()
    U = Union(R, C)
    assert U.domain_1 == R
    assert U.domain_2 == C   
    U = Union(C, R)
    assert U.domain_1 == R
    assert U.domain_2 == C 


def test_cant_unite_non_disjoint_intervals():
    I1 = Interval(0,1)
    I2 = Interval(0,0.5)
    with pytest.raises(ValueError):
        _ = Union(I1, I2)
    with pytest.raises(ValueError):
        _ = Union(I2, I1)


def test_can_unite_disjoint_intervals():
    I1 = Interval(0,1)
    I2 = Interval(3,5)
    _ = Union(I1, I2)


def test_cant_unite_different_dim():
    I1 = Interval(0,1)
    R = Rectangle([0, 0], [0, 1], [1, 0])
    with pytest.raises(ValueError):
        _ = Union(R, I1)


def test_union_new_dim():
    R, C = _create_domains()
    U = Union(R, C)
    assert U.dim == 2


def test_union_tol():
    R, C = _create_domains()
    U = Union(R, C)
    assert U.tol == R.tol
    C.tol = 1e-08
    U = Union(R, C)
    assert U.tol == C.tol


def test_union_surface_approxi():
    R, C = _create_domains()
    U = Union(R, C)
    assert np.isclose(U.surface, 3+3*np.pi/4, atol=1e-01)
    U = Union(R, C, n=10000)
    assert np.isclose(U.surface, 3+3*np.pi/4, atol=1e-03)


def test_union_volume_approxi():
    R, C = _create_domains()
    U = Union(R, C)
    assert np.isclose(U.volume, 1+3/4*np.pi*0.5**2, atol=1e-01)


def test_union_points_inside_and_outside():
    R, C = _create_domains()
    U = Union(R, C)
    points = R.sample_inside(50)
    inside = U.is_inside(points)
    assert all(inside)
    points = C.sample_inside(50)
    inside = U.is_inside(points)
    assert all(inside)
    points = [[1.2, 0], [-1, -1], [1, 1.01]]
    inside = U.is_inside(points)
    assert not any(inside)   


def test_union_on_boundary():
    R, C = _create_domains()
    U = Union(R, C)    
    points = [[0, 0], [0.2, 0], [0, 0.2], [0.5*np.cos(np.pi/4), 0.5*np.sin(np.pi/4)],
              [0, 0.51], [1, 1], [1, 0.3],
              [-0.5*np.cos(np.pi/4), -0.5*np.sin(np.pi/4)]]
    on_bound = U.is_on_boundary(points)
    assert not any(on_bound[0:4])
    assert all(on_bound[4:])


def test_random_sampling_inside_union():
    R, C = _create_domains()
    U = Union(R, C)    
    points = U.sample_inside(421)
    assert np.shape(points) == (421, 2)
    assert U.is_inside(points).all()
    in_c = C.is_inside(points)
    in_r = R.is_inside(points)
    assert np.logical_or(in_c, in_r).all()


def test_grid_sampling_inside_union():
    R, C = _create_domains()
    U = Union(R, C)    
    points = U.sample_inside(423, type='grid')
    assert np.shape(points) == (423, 2)
    assert U.is_inside(points).all()
    in_c = C.is_inside(points)
    in_r = R.is_inside(points)
    assert np.logical_or(in_c, in_r).all()


def test_lhs_sampling_in_union_of_interval():
    U = Union(Interval(0, 1), Interval(3, 5))
    points = U.sample_inside(42, type='lhs')
    assert np.shape(points) == (42, 1)
    assert U.is_inside(points).all()    


def test_random_sampling_boundary_union():
    R, C = _create_domains()
    U = Union(R, C)    
    points = U.sample_boundary(10)
    assert np.shape(points) == (10, 2)
    assert U.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all()


def test_grid_sampling_boundary_union():
    R, C = _create_domains()
    U = Union(R, C)    
    points = U.sample_boundary(30, type='grid')
    assert np.shape(points) == (30, 2)
    assert U.is_on_boundary(points).all()
    on_c = C.is_on_boundary(points)
    on_r = R.is_on_boundary(points)
    assert np.logical_or(on_c, on_r).all()


def test_normal_sampling_boundary_union():
    R, C = _create_domains()
    U = Union(R, C)    
    points = U.sample_boundary(30, type='normal', sample_params={'mean': [0.5, 0],
                                                                 'cov': 0.1})
    assert np.shape(points) == (30, 2)
    assert U.is_on_boundary(points).all()


def test_grid_for_plots_union():
    R, C = _create_domains()
    U = Union(R, C)
    points = U.grid_for_plots(200)
    inside = U.is_inside(points)
    on_bound = U.is_on_boundary(points)
    assert np.logical_or(inside, on_bound).all()   


def test_normals_of_union():
    R, C = _create_domains()
    U = Union(R, C)
    points = [[1, 0.5], [0.3, 1], [0, 0.7], 
              [-0.5*np.cos(np.pi/4), -0.5*np.sin(np.pi/4)]]  
    normals = U.boundary_normal(points)
    assert np.allclose(normals[0], [1, 0]) 
    assert np.allclose(normals[1], [0, 1]) 
    assert np.allclose(normals[2], [-1, 0]) 
    assert np.allclose(normals[3], [-np.cos(np.pi/4), -np.sin(np.pi/4)]) 
    point = [[0.6, 0]]
    normal = U.boundary_normal(point)
    assert np.allclose(normal[0], [0, -1])  
    point = [[0.5*np.cos(3*np.pi/4), 0.5*np.sin(3*np.pi/4)]]
    normal = U.boundary_normal(point)
    assert np.allclose(normal[0], [np.cos(3*np.pi/4), np.sin(3*np.pi/4)])  

def test_unite_many_times():
    R, C = _create_domains()
    U = Union(R, C)
    U = Union(U, Circle([1, 1], 0.5))
    U = Union(U, Circle([0, 1], 0.5))
    b = U.sample_inside(20)
    i = U.sample_boundary(20)
    assert U.is_inside(b).all()
    assert U.is_on_boundary(i).all()


def test_outline_union():
    R = Rectangle([0, 0], [2, 0], [0, 2])
    R2 = Rectangle([0.5, 0.5], [1, 0.5], [0.5, 1])
    C = Cut(R, R2)
    T = Triangle([0, 2], [1, 2], [0, 3])
    U = Union(C, T)
    outline = U.outline()
    assert isinstance(outline, list)
    assert np.allclose(outline[0], [[0, 0], [0, 2], [0, 3],
                                    [1, 2], [2, 2], [2, 0], [0, 0]])
    assert np.allclose(outline[1], [[0.5, 0.5], [1, 0.5], [1, 1],
                                    [0.5, 1], [0.5, 0.5]])


def _create_intervals():
    I1 = Interval(0, 1)
    I2 = Interval(2, 4)
    return I1, I2


def test_union_of_intervals_volume_and_surface():
    I1, I2 = _create_intervals()
    U = Union(I1, I2)
    assert U.volume == 3
    assert U.surface == 4


def test_union_of_intervals_grid_sampling():
    I1, I2 = _create_intervals()
    U = Union(I1, I2)
    points = U.sample_inside(20, type='grid')
    assert I1.is_inside(points[13:20]).all()   
    assert I2.is_inside(points[0:13]).all()  


def test_union_of_intervals_grid_sampling_boundary():
    I1, I2 = _create_intervals()
    U = Union(I1, I2)
    points = U.sample_boundary(20, type='grid')
    assert all(points[0:5] == 2)   
    assert all(points[5:10] == 4)   
    assert all(points[10:15] == 0)   
    assert all(points[15:20] == 1)   


def test_union_of_intervals_lower_bound_sampling():
    I1, I2 = _create_intervals()
    U = Union(I1, I2)
    points = U.sample_boundary(20, type='lower_bound_only')
    assert all(points == 0)    


def test_union_of_intervals_upper_bound_sampling():
    I1, I2 = _create_intervals()
    U = Union(I1, I2)
    points = U.sample_boundary(20, type='upper_bound_only')
    assert all(points == 4)  


def test_serialize_union():
    R, C = _create_domains()
    U = Union(R, C)
    dct = U.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == '(Rectangle + Circle)'
    U = Cut(U, Circle([1, 1], 0.5))
    dct = U.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == '((Rectangle + Circle) - Circle)'


def test_union_bounds():
    R, C = _create_domains()
    U = Union(R, C)
    bounds = U._compute_bounds()
    assert np.allclose(bounds, [-0.5, 1, -0.5, 1])


# Test intersection

def test_intersection():
    R, C = _create_domains()
    _ = Intersection(R, C)


def test_cant_intersect_intervals():
    I1 = Interval(0,1)
    I2 = Interval(0,0.5)
    with pytest.raises(ValueError):
        _ = Intersection(I1, I2)


def test_cant_intersect_different_dim():
    I1 = Interval(0,1)
    R = Rectangle([0, 0], [0, 1], [1, 0])
    with pytest.raises(ValueError):
        _ = Intersection(R, I1)


def test_intersection_dim():
    R, C = _create_domains()
    I = Intersection(R, C)
    assert I.dim == 2


def test_intersection_tol():
    R, C = _create_domains()
    I = Intersection(R, C)
    assert I.tol == R.tol


def test_intersection_surface_approxi():
    R, C = _create_domains()
    I = Intersection(R, C)
    assert np.isclose(I.surface, 1+np.pi/4, atol=1e-01)
    I = Intersection(R, C, n=10000)
    assert np.isclose(I.surface, 1+np.pi/4, atol=1e-03)


def test_intersection_volume_approxi():
    R, C = _create_domains()
    I = Intersection(R, C)
    assert np.isclose(I.volume, np.pi/4*0.5**2, atol=1e-01)
    I = Intersection(C, R)
    assert np.isclose(I.volume, np.pi/4*0.5**2, atol=1e-01)


def test_intersection_points_inside_and_outside():
    R, C = _create_domains()
    I = Intersection(R, C)
    points = [[-0.001, 0], [0, 0.501], [0.5, 0.5], [0.1, 0.1], [0.25, 0.1]]
    inside = I.is_inside(points)
    assert not any(inside[0:3])
    assert all(inside[3:5])


def test_intersection_on_boundary():
    R, C = _create_domains()
    I = Intersection(R, C)    
    points = [[0.5*np.cos(np.pi/4), 0.5*np.cos(np.pi/4)], [0.25, 0], [0, 0],
              [1, 1], [0, 0.51]]
    on_bound = I.is_on_boundary(points)
    assert all(on_bound[0:3])
    assert not any(on_bound[3:5])


def test_random_sampling_inside_intersection():
    R, C = _create_domains()
    I = Intersection(R, C)    
    points = I.sample_inside(50)
    assert np.shape(points) == (50, 2)
    assert I.is_inside(points).all()
    assert R.is_inside(points).all()
    assert C.is_inside(points).all()


def test_grid_sampling_inside_intersection():
    R, C = _create_domains()
    I = Intersection(R, C)    
    points = I.sample_inside(150, type='grid')    
    assert np.shape(points) == (150, 2)
    assert I.is_inside(points).all()
    assert R.is_inside(points).all()
    assert C.is_inside(points).all()
    points = I.sample_inside(50, type='grid')    
    assert np.shape(points) == (50, 2)
    assert I.is_inside(points).all()
    assert R.is_inside(points).all()
    assert C.is_inside(points).all()


def test_random_sampling_boundary_intersection():
    R, C = _create_domains()
    I = Intersection(R, C)    
    points = I.sample_boundary(32)    
    assert np.shape(points) == (32, 2)
    assert I.is_on_boundary(points).all()
    on_R = R.is_on_boundary(points)
    on_C = C.is_on_boundary(points)
    assert np.logical_or(on_R, on_C).all()    


def test_grid_sampling_boundary_intersection():
    R, C = _create_domains()
    I = Intersection(R, C)    
    points = I.sample_boundary(24, type='grid')    
    assert np.shape(points) == (24, 2)
    assert I.is_on_boundary(points).all()
    on_R = R.is_on_boundary(points)
    on_C = C.is_on_boundary(points)
    assert np.logical_or(on_R, on_C).all()  


def test_normal_sampling_boundary_intersection():
    R, C = _create_domains()
    U = Intersection(R, C)    
    mean = [0.5*np.cos(0.01), 0.5*np.sin(0.01)]
    points = U.sample_boundary(30, type='normal', sample_params={'mean': mean,
                                                                 'cov': 0.2})
    assert np.shape(points) == (30, 2)
    assert U.is_on_boundary(points).all()


def test_boundary_normal_intersection():
    R, C = _create_domains()
    I = Intersection(R, C) 
    points = [[0, 0], [0.3, 0], [0, 0.2], 
              [0.5*np.cos(np.pi/4), 0.5*np.sin(np.pi/4)]]
    normals = I.boundary_normal(points)
    assert np.allclose(normals[0], [-1/np.sqrt(2), -1/np.sqrt(2)])
    assert np.equal(normals[1], [0, -1]).all()
    assert np.equal(normals[2], [-1, 0]).all()
    assert np.allclose(normals[3], [np.cos(np.pi/4), np.sin(np.pi/4)]) 


def test_serialize_intersection():
    R, C = _create_domains()
    I = Intersection(R, C) 
    dct = I.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == '(Rectangle intersect Circle)'


def test_grid_for_plots_intersection():
    R, C = _create_domains()
    U = Intersection(R, C)
    points = U.grid_for_plots(200)
    inside = U.is_inside(points)
    on_bound = U.is_on_boundary(points)
    assert np.logical_or(inside, on_bound).all()  


def test_intersection_bounds():
    R, C = _create_domains()
    I = Intersection(R, C)
    bounds = I._compute_bounds()
    assert np.allclose(bounds, [0, 0.5, 0, 0.5])


def test_outline_intersection():
    R = Polygon2D(corners=[[0, 0], [2, 0], [2, 2], [0, 2]])
    R2 = Rectangle([0.5, 0.5], [0.7, 0.5], [0.5, 0.7])
    C = Cut(R, R2)
    Ci = Circle([0, 0], 2)
    I = Intersection(C, Ci)
    outline = I.outline()
    assert isinstance(outline, list)
    assert np.allclose(outline[1], [[0.5, 0.5], [0.7, 0.5], [0.7, 0.7],
                                    [0.5, 0.7], [0.5, 0.5]])