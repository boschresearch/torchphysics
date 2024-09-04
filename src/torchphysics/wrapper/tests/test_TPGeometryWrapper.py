import pytest
import torchphysics as tp
import numpy as np
import torch

from torchphysics.wrapper.geometry import TPGeometryWrapper
from torchphysics.problem.domains import (Interval, Circle, Sphere, Triangle, Parallelogram)
from torchphysics.problem.domains.domain2D.shapely_polygon import ShapelyPolygon
from torchphysics.problem.domains.domain3D.trimesh_polyhedron import TrimeshPolyhedron
from modulus.sym.geometry.primitives_1d import (Line1D)
from modulus.sym.geometry.primitives_2d import Circle as Circle_modulus
from modulus.sym.geometry.primitives_2d import Triangle as Triangle_modulus
from modulus.sym.geometry.primitives_2d import Polygon
from modulus.sym.geometry.primitives_3d import Sphere as Sphere_modulus

from sympy import Eq, Symbol



def single_domain_dict(domain):    
    domain_dict = {'d': domain}
    return domain_dict

def union_domain_dict(domain1,domain2):    
    domain_dict = {'u': [{'d': domain1},{'d': domain2}]}
    return domain_dict

def intersection_domain_dict(domain1,domain2):
    return {'i': [{'d': domain1},{'d': domain2}]}


def cut_domain_dict(domain1,domain2):
    return {'c' : [{'d': domain1},{'d': domain2}]}   

def translate_domain_dict(domain,translation_params):    
    return {'t' : [{'d': domain},translation_params]}

def rotate_domain_dict(domain,rotation_matrix,rotation_center):    
    return {'r' : [{'d': domain},rotation_matrix,rotation_center]}

def combined_operations_dict(domain1,domain2,domain3,translation_params,rotation_matrix,rotation_center):    
    return {'i' : [{'t': [{'r' :  [{'d': domain1},rotation_matrix,rotation_center]},translation_params]},{'p':[{'d': domain2},{'d': domain3}]}]}


def product_domain_dict(domain1,domain2):    
    return {'p': [{'d': domain1},{'d': domain2}]}

# Fixtures for different domain types
@pytest.fixture
def interval_domain():
    X = tp.spaces.R1('x')    
    return Interval(X,0, 1)

@pytest.fixture
def interval_domain2():
    X = tp.spaces.R1('x')    
    return Interval(X,2,4)

@pytest.fixture
def interval_domain_y():
    Y = tp.spaces.R1('y')    
    return Interval(Y,0,4)


@pytest.fixture
def interval_domain_z():
    Z = tp.spaces.R1('z')    
    return Interval(Z,0,4)



@pytest.fixture
def circle_domain():
    X = tp.spaces.R2('x')    
    return Circle(X,center=(0, 0), radius=1)

@pytest.fixture
def circle_domain_xy():
    X = tp.spaces.R1('x')    
    Y = tp.spaces.R1('y')
    return Circle(X*Y,center=(0, 0), radius=1)    

@pytest.fixture
def sphere_domain():
    X = tp.spaces.R3('x') 
    return Sphere(X,center=(0, 0, 0), radius=1)

@pytest.fixture
def triangle_domain():
    X = tp.spaces.R2('x')
    return Triangle(X,origin=(0, 0), corner_1=(1, 0), corner_2=(0.5, np.sqrt(3)/2))

@pytest.fixture
def parallelogram_domain():
    X = tp.spaces.R2('x')
    return Parallelogram(X,origin=(0, 0), corner_1=(1, 0), corner_2=(0.5, 0.5))

@pytest.fixture
def parallelogram_domain_xy():
    XY = tp.spaces.R1('x')*tp.spaces.R1('y')
    return Parallelogram(XY,origin=(0, 0), corner_1=(1, 0), corner_2=(0, 2))

@pytest.fixture
def trimesh_polyhedron_domain():        
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, -1]]
    faces = [[0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 2, 4], [0, 1, 4], [1, 2, 4]]
    poly3D = TrimeshPolyhedron(tp.spaces.R3('x'), vertices=vertices, faces=faces)          
    return poly3D

@pytest.fixture 
def shapely_polygon_domain():
    X = tp.spaces.R2('x')
    return ShapelyPolygon(X, vertices=[[0, 0], [1, 0], [1, 2], [0, 1]])

# Test functions for each domain type
def test_interval_domain(interval_domain):    
    is_boundary, geometry, cond, _, _= TPGeometryWrapper(single_domain_dict(interval_domain)).getModulusGeometry()            
    assert isinstance(geometry, Line1D)  
    assert not is_boundary
    assert cond == None
   
def test_circle_domain(circle_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(circle_domain)).getModulusGeometry()            
    assert isinstance(geometry, Circle_modulus)  
    assert not is_boundary
    assert cond == None

def test_sphere_domain(sphere_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(sphere_domain)).getModulusGeometry()            
    assert isinstance(geometry, Sphere_modulus)  
    assert not is_boundary
    assert cond == None

def test_triangle_domain(triangle_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(triangle_domain)).getModulusGeometry()            
    assert isinstance(geometry, Triangle_modulus)  
    assert not is_boundary
    assert cond == None

def test_parallelogram_domain(parallelogram_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(parallelogram_domain)).getModulusGeometry()            
    assert isinstance(geometry, Polygon)      
    assert not is_boundary
    assert cond == None

def test_trimesh_polyhedron_domain(trimesh_polyhedron_domain):  
    with pytest.raises(Exception) as excinfo: 
        is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(trimesh_polyhedron_domain)).getModulusGeometry()            
    try:        
        assert isinstance(geometry, Tessellation)
        assert not is_boundary
        assert cond == None
    except NameError:
        assert "Tessellation module only supported for Modulus docker installation due to missing pysdf installation!" in str(excinfo.value)
    

def test_boundary_of_interval_domain(interval_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(interval_domain.boundary_left)).getModulusGeometry()            
    assert isinstance(geometry, Line1D)  
    assert is_boundary
    assert (cond==Eq(Symbol('x'),0))

def test_boundary_of_circle_domain(circle_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(circle_domain.boundary)).getModulusGeometry()
    assert isinstance(geometry, Circle_modulus)  
    assert is_boundary

def test_boundary_of_sphere_domain(sphere_domain):   
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(sphere_domain.boundary)).getModulusGeometry()
    assert isinstance(geometry, Sphere_modulus)  
    assert is_boundary

def test_boundary_of_triangle_domain(triangle_domain):    
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(triangle_domain.boundary)).getModulusGeometry()
    assert isinstance(geometry,Triangle_modulus)  
    assert is_boundary

def test_boundary_of_parallelogram_domain(parallelogram_domain):   
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(parallelogram_domain.boundary)).getModulusGeometry()
    assert isinstance(geometry, Polygon)  
    assert is_boundary

def test_boundary_of_trimesh_polyhedron_domain(trimesh_polyhedron_domain):       
    with pytest.raises(Exception) as excinfo: 
        is_boundary, geometry, cond, _, _ = TPGeometryWrapper(single_domain_dict(trimesh_polyhedron_domain.boundary)).getModulusGeometry()            
    try:        
        assert isinstance(geometry, Tesselation)
        assert is_boundary
        assert cond == None
    except NameError:
        assert "Tessellation module only supported for Modulus docker installation due to missing pysdf installation!" in str(excinfo.value)




def test_union_operation(interval_domain,interval_domain2):      
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(union_domain_dict(interval_domain,interval_domain2)).getModulusGeometry()            
    assert not is_boundary    
    assert geometry.sdf({'x': [[1.5]]},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': [[3]]},params={})['sdf'][0][0] > 0
    assert geometry.sdf({'x': [[0.5]]},params={})['sdf'][0][0]>0
    assert abs(geometry.sdf({'x': [[2]]},params={})['sdf'][0][0]) < 1e-12
    assert cond == None
    


def test_intersection_operation(circle_domain,parallelogram_domain):
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(intersection_domain_dict(circle_domain,parallelogram_domain)).getModulusGeometry()            
    assert not is_boundary    
    assert abs(geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[0.5]])},params={})['sdf'][0][0]) < 1e-16
    assert geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[0.1]])},params={})['sdf'][0][0] > 0
    assert geometry.sdf({'x': np.array([[2]]),'y':np.array([[0]])},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': np.array([[-0.5]]),'y':np.array([[0.5]])},params={})['sdf'][0][0] < 0
    assert cond == None

def test_cut_operation(circle_domain,parallelogram_domain):
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(cut_domain_dict(circle_domain,parallelogram_domain)).getModulusGeometry()            
    assert not is_boundary    
    assert abs(geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[0.5]])},params={})['sdf'][0][0]) < 1e-16
    assert geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[0.1]])},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': np.array([[2]]),'y':np.array([[0]])},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': np.array([[-0.5]]),'y':np.array([[0.5]])},params={})['sdf'][0][0] > 0
    assert cond == None

def test_translate_operation(sphere_domain):  
    is_boundary, geometry, cond, translate_vec, rotation_list = TPGeometryWrapper(translate_domain_dict(sphere_domain,torch.tensor([5,0,0]))).getModulusGeometry() 
    assert not is_boundary
    assert cond == None
    assert rotation_list == [[],[],[],1]
    assert translate_vec == [0,0,0]
    assert geometry.sdf({'x': np.array([[0]]),'y':np.array([[0]]),'z':np.array([[0]])},params={})['sdf'][0][0] < 0
    assert abs(geometry.sdf({'x': np.array([[4]]),'y':np.array([[0]]),'z':np.array([[0]])},params={})['sdf'][0][0]) < 1e-12
    assert geometry.sdf({'x': np.array([[5.5]]),'y':np.array([[0.5]]),'z':np.array([[-0.5]])},params={})['sdf'][0][0] > 0
    
    
def test_rotate_operation(circle_domain):  
    is_boundary, geometry, cond, translate_vec, rotation_list = TPGeometryWrapper(rotate_domain_dict(circle_domain,torch.tensor([[[-1.0000e+00,  0],
         [0, -1.0000e+00]]]),torch.tensor([5,0]))).getModulusGeometry() 
    assert not is_boundary
    assert cond == None
    assert rotation_list == [[],[],[],0]
    assert translate_vec == [0,0,0]
    assert geometry.sdf({'x': np.array([[12]]),'y':np.array([[0]])},params={})['sdf'][0][0] < 0
    assert abs(geometry.sdf({'x': np.array([[11]]),'y':np.array([[0]])},params={})['sdf'][0][0]) <1e-12
    assert geometry.sdf({'x': np.array([[10]]),'y':np.array([[0.5]])},params={})['sdf'][0][0] > 0


def test_product_operation(parallelogram_domain_xy,interval_domain_z):
    is_boundary, geometry, cond, _, _ = TPGeometryWrapper(product_domain_dict(parallelogram_domain_xy,interval_domain_z)).getModulusGeometry()            
    assert not is_boundary    
    assert geometry.sdf({'x': np.array([[2]]),'y':np.array([[1]]),'z':np.array([[1]])},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[1]]),'z':np.array([[2]])},params={})['sdf'][0][0] > 0
    assert abs(geometry.sdf({'x': np.array([[1]]),'y':np.array([[1]]),'z':np.array([[1]])},params={})['sdf'][0][0]) <1e-12
     
    assert cond == None


def test_multiple_operations(circle_domain_xy,interval_domain,interval_domain_y):
    is_boundary, geometry, cond, translate_vec, rotation_list = TPGeometryWrapper(combined_operations_dict(circle_domain_xy,interval_domain,interval_domain_y,torch.tensor([1,-1]),torch.tensor([[[-1.0000e+00,  0],[0, -1.0000e+00]]]),torch.tensor([0,2]))).getModulusGeometry() 
    assert not is_boundary
    assert cond == None
    assert rotation_list == [[],[],[],0]
    assert translate_vec == [0,0,0]
    # left half circle with radius 1 and center at (1,3) 
    assert geometry.sdf({'x': np.array([[1.5]]),'y':np.array([[3]])},params={})['sdf'][0][0] < 0
    assert geometry.sdf({'x': np.array([[0.5]]),'y':np.array([[3]])},params={})['sdf'][0][0] > 0
    assert abs(geometry.sdf({'x': np.array([[1]]),'y':np.array([[3]])},params={})['sdf'][0][0]) <1e-12


