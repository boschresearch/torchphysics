import pytest
import numpy as np
import os

from torchphysics.problem.domain.domain3D import (Box, 
                                                  Sphere,
                                                  Cylinder,
                                                  Polygon3D)


# Test Box
def test_create_box():
    B = Box([0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 4])
    assert np.allclose(B.corner_o, [0, 0, 0])
    assert np.allclose(B.corner_x, [1, 0, 0])
    assert np.allclose(B.corner_y, [0, 2, 0])
    assert np.allclose(B.corner_z, [0, 0, 4])
    assert np.isclose(B.tol, 1e-06)
    assert B.dim == 3
    assert np.isclose(B.surface, 28)
    assert np.isclose(B.volume, 8)
    normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    assert np.allclose(B.normals, normals)
    assert np.allclose(B.side_lengths, [1, 2, 4])


def test_create_rotated_box():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    assert np.allclose(B.corner_o, [1, 0, 0])
    assert np.allclose(B.corner_x, [2, 1, 0])
    assert np.allclose(B.corner_y, [0, 1, 0])
    assert np.allclose(B.corner_z, [1, 0, 1])
    assert np.allclose(B.side_lengths, [np.sqrt(2), np.sqrt(2), 1])
    assert np.isclose(B.surface, 4+4*np.sqrt(2))
    assert np.isclose(B.volume, 2)  
    normals = [[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0]]
    assert np.allclose(B.normals[0:4], normals/np.sqrt(2)) 
    assert np.allclose(B.inverse_matrix, [[1/2 , 1/2, 0], 
                                          [-1/2, 1/2, 0], 
                                          [0   , 0  , 1]])


def test_create_wrong_box():
    with pytest.raises(ValueError):
        _ = Box([1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1])


def test_points_inside_box():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = [[-1, 0, 0], [1, 1, 0], [1, 1, 0.5], [1, 1, 1.1]]
    inside = B.is_inside(points)
    assert np.shape(inside) == (4, 1)
    assert not inside[0]
    assert inside[1] and inside[2]
    assert not inside[3]


def test_points_on_boundary_box():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = [[-1, 0, 0], [1, 1, 0.5], 
              [1, 1, 0], [1, 1, 1], [1.5, 0.5, 0.1], 
              [1.5, 1.5, 0], [0.7, 1.7, 0.8], [0.5, 0.5, 0.7]]
    bound = B.is_on_boundary(points)
    assert np.shape(bound) == (8, 1)
    assert not bound[0]
    assert not bound[1]
    assert np.all(bound[2:8])


def test_random_sampling_inside_box():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = B.sample_inside(150, type='random')
    assert np.shape(points) == (150, 3)
    assert np.all(B.is_inside(points))


def test_grid_sampling_inside_box():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = B.sample_inside(50, type='grid')
    assert np.shape(points) == (50, 3)
    assert np.all(B.is_inside(points))
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    points = B.sample_inside(27, type='grid')
    assert np.shape(points) == (27, 3)
    assert np.all(B.is_inside(points))


def test_random_sampling_on_box_boundary():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = B.sample_boundary(50, type='random')
    assert np.shape(points) == (50, 3)
    assert np.all(B.is_on_boundary(points))
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    points = B.sample_boundary(3, type='random')
    assert np.shape(points) == (3, 3)
    assert np.all(B.is_on_boundary(points))


def test_grid_sampling_on_box_boundary():
    B = Box([1, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0, 1])
    points = B.sample_boundary(50, type='grid')
    assert np.shape(points) == (50, 3)
    assert np.all(B.is_on_boundary(points))
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    points = B.sample_boundary(54, type='grid')
    assert np.shape(points) == (54, 3)
    assert np.all(B.is_on_boundary(points))


def test_output_type_box():
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    i_rand = B.sample_inside(1)
    b_rand = B.sample_boundary(5)
    i_grid = B.sample_inside(5, type='grid')
    b_grid = B.sample_boundary(6, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)

def test_serialize_box():
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    dct = B.serialize()
    assert dct['dim'] == 3
    assert dct['tol'] == 1e-06
    assert dct['name'] == 'Box'
    assert np.equal(dct['origin'], [0, 0, 0]).all()
    assert np.equal(dct['x-corner'], [1, 0, 0]).all()
    assert np.equal(dct['y-corner'], [0, 1, 0]).all()
    assert np.equal(dct['z-corner'], [0, 0, 1]).all()


def test_bounds_of_box():
    B = Box([0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 1])
    bounds = B._compute_bounds()
    assert np.allclose(bounds, [0, 2, 0, 4, 0, 1])


def test_boundary_normals_box():
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2])
    points = [[0.3, 0.3, 0], [0.3, 0.3, 2], [1, 0.4, 0.4], 
              [0, 0.4, 0.3], [0.7, 0, 0.8], [0.4, 1, 0.7]]
    normals = B.boundary_normal(points)
    assert np.shape(normals) == (6, 3)
    correct_normals = [[0, 0, -1], [0, 0, 1], [1, 0, 0],
                       [-1, 0, 0], [0, -1, 0], [0, 1, 0]]
    assert np.allclose(correct_normals, normals)


def test_boundary_normals_box_with_corner_and_edge_point():
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2])
    points = [[0, 0, 0], [0.3, 1, 2]]
    normals = B.boundary_normal(points)
    assert np.shape(normals) == (2, 3)
    r_3 = np.sqrt(3)
    r_2 = np.sqrt(2)
    correct_normals = [[-1/r_3, -1/r_3, -1/r_3], [0, 1/r_2, 1/r_2]]
    assert np.allclose(correct_normals, normals)


def test_console_output_when_not_on_bound_box(capfd):
    B = Box([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2])
    point = [[0.5, 0.5, -10]]
    B.boundary_normal(point)
    out, _ = capfd.readouterr() 
    assert out == 'Warning: some points are not at the boundary!\n'


# Test Sphere
def test_create_sphere():
    S = Sphere(center=[1, 2, 3], radius=3)
    assert np.allclose(S.center, [1, 2, 3])
    assert np.allclose(S.radius, 3)
    assert np.isclose(S.tol, 1e-06)
    assert S.dim == 3
    assert np.isclose(S.surface, 4*np.pi*9)
    assert np.isclose(S.volume, 4/3*np.pi*9)


def test_points_inside_sphere():
    S = Sphere(center=[1, 0, 0], radius=3)
    p = [[0, 0, 0], [10, 30, 2], [2.5, 0.2, 0]]
    inside = S.is_inside(p)
    assert np.shape(inside) == (3, 1)
    assert inside[0]
    assert not inside[1]
    assert inside[2]


def test_points_on_sphere_boundary():
    S = Sphere(center=[1, 2, 0], radius=2)
    p = [[0, 0, 0], [10, 30, 2], [3, 2, 0]]
    bound = S.is_on_boundary(p)
    assert np.shape(bound) == (3, 1)
    assert not bound[0]
    assert not bound[1]
    assert bound[2]


def test_get_bounds_sphere():
    S = Sphere(center=[1, 2, 0], radius=2)
    bounds = S._compute_bounds()
    assert np.allclose(bounds, [-1, 3, 0, 4, -2, 2])


def test_random_sampling_inside_sphere():
    S = Sphere(center=[1, 2, 0], radius=2)
    points = S.sample_inside(30, type='random')
    assert np.shape(points) == (30, 3)
    assert np.all(S.is_inside(points))


def test_grid_sampling_inside_sphere():
    S = Sphere(center=[1, 2, 0], radius=2)
    points = S.sample_inside(30, type='grid')
    assert np.shape(points) == (30, 3)
    assert np.all(S.is_inside(points))
    S = Sphere([1 ,1 , 1], 2)
    points = S.sample_inside(500, type='grid')
    assert np.shape(points) == (500, 3)
    assert np.all(S.is_inside(points))


def test_random_sampling_boundary_sphere():
    S = Sphere(center=[1, 2, 0], radius=2)
    points = S.sample_boundary(20, type='random')
    assert np.shape(points) == (20, 3)
    assert np.all(S.is_on_boundary(points))


def test_grid_sampling_boundary_sphere():
    S = Sphere(center=[1, 2, 0], radius=3)
    points = S.sample_boundary(20, type='grid')
    assert np.shape(points) == (20, 3)
    assert np.all(S.is_on_boundary(points))


def test_output_type_sphere():
    S = Sphere(center=[1, 2, 0], radius=1)
    i_rand = S.sample_inside(1)
    b_rand = S.sample_boundary(1)
    i_grid = S.sample_inside(1, type='grid')
    b_grid = S.sample_boundary(2, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_serialize_sphere():
    S = Sphere(center=[1, 2, 0], radius=3, tol=0.1)
    dct = S.serialize()
    assert dct['dim'] == 3
    assert dct['tol'] == 0.1
    assert dct['name'] == 'Sphere'
    assert np.equal(dct['center'], [1, 2, 0]).all()
    assert np.equal(dct['radius'], 3).all()


def test_normals_sphere():
    S = Sphere(center=[1, 2, 0], radius=3)
    points = S.sample_boundary(15, type='grid')
    normals = S.boundary_normal(points)
    assert np.shape(normals) == (15,3)
    for i in range(15):
        assert np.isclose(np.linalg.norm(normals[i]), 1)


def test_console_output_when_not_on_bound_sphere(capfd):
    S = Sphere(center=[1, 2, 0], radius=3)
    point = [[0.1, 0.1, 10]]
    S.boundary_normal(point)
    out, _ = capfd.readouterr() 
    assert out == 'Warning: some points are not at the boundary!\n'


# Test Cylinder

def test_create_cylinder():
    C = Cylinder(center=[0, 0, 0], radius=2, height=4, orientation=[0, 0, 1])
    assert C.height == 4
    assert C.radius == 2
    assert np.allclose(C.center, [0, 0, 0])
    assert np.allclose(C.orientation, [0, 0, 1])
    assert np.allclose(C.rotation_matrix, np.eye(3))


def test_create_orientated_cylinder():
    C = Cylinder(center=[0, 2, 0], radius=3, height=3, orientation=[1, 0, 0])
    assert C.height == 3
    assert C.radius == 3
    assert np.allclose(C.center, [0, 2, 0])
    assert np.allclose(C.orientation, [1, 0, 0])
    assert np.allclose(C.rotation_matrix, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    C = Cylinder(center=[0, 2, 0], radius=3, height=3, orientation=[0, 0, -1])
    assert np.allclose(C.rotation_matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C = Cylinder(center=[0, 2, 0], radius=2, height=4, orientation=[1, 1, 0])
    assert np.allclose(C.rotation_matrix, [[0.5, -0.5, 0.707],
                                           [-0.5, 0.5, 0.707],
                                           [-0.707, -0.707, 0]], atol=0.01)


def test_points_inside_cylinder():
    C = Cylinder(center=[0, 2, 0], radius=3, height=3, orientation=[1, 0, 0])
    points = [[0, 2, 0], [1.5, 2, 0], [-1, 2, 0], [0, 3, 1], 
              [0, -2, 0], [10, 24, 32]]
    inside = C.is_inside(points)
    assert np.shape(inside) == (6, 1)
    assert np.all(inside[0:4])
    assert not np.any(inside[4:6])


def test_points_on_cylinder_boundary():
    C = Cylinder(center=[0, 2, 0], radius=3, height=3, orientation=[1, 0, 0])
    points = [[0, 2, 0], [2, 2, 0], [-1, 2, 0], [0, 3, 1], 
              [0, -2, 0], [10, 24, 32], [-1.5, 2, 1], [1.5, 3, 0], 
              [1, 5, 0], [-1, 2, 3]]
    inside = C.is_on_boundary(points)
    assert np.shape(inside) == (10, 1)
    assert not np.any(inside[0:6])
    assert np.all(inside[6:10])


def test_random_sampling_inside_cylinder():
    C = Cylinder(center=[0, 0, 0], radius=3, height=3, orientation=[0, 0, 1])
    points = C.sample_inside(50, type='random')
    assert np.shape(points) == (50, 3)
    assert np.all(C.is_inside(points))


def test_grid_sampling_inside_cylinder():
    C = Cylinder(center=[0, 0, 0], radius=3, height=3, orientation=[0, 0, 1])
    points = C.sample_inside(50, type='grid')
    assert np.shape(points) == (50, 3)
    assert np.all(C.is_inside(points))
    points = C.sample_inside(1, type='grid')
    assert np.shape(points) == (1, 3)
    assert np.allclose(points, [0, 0, 0])


def test_random_sampling_boundary_cylinder():
    C = Cylinder(center=[1, 0, 1], radius=5, height=3, orientation=[0, 1, 1])
    points = C.sample_boundary(20, type='random')
    assert np.shape(points) == (20, 3)
    assert np.all(C.is_on_boundary(points)) 


def test_grid_sampling_boundary_cylinder():
    C = Cylinder(center=[0, 3, 0], radius=5, height=3, orientation=[0, 1, 1])
    points = C.sample_boundary(20, type='grid')
    assert np.shape(points) == (20, 3)
    assert np.all(C.is_on_boundary(points)) 


def test_boundary_normals_cylinder():
    C = Cylinder(center=[0, 0, 1], radius=1, height=2, orientation=[0, 1, 1])
    sqrt_2 = np.sqrt(2)
    p = [[0, 1/sqrt_2, 1+1/sqrt_2], [0, -1/sqrt_2, 1-1/sqrt_2],
         [1, 0, 1], [0, 1/sqrt_2, 1-1/sqrt_2]]
    normals = C.boundary_normal(p)
    assert np.shape(normals) == (4, 3)
    correct_normals = [[0, 1/sqrt_2, 1/sqrt_2], [0, -1/sqrt_2, -1/sqrt_2], 
                       [1, 0, 0], [0, 1/sqrt_2, -1/sqrt_2]]
    assert np.allclose(normals, correct_normals)


def test_console_output_when_not_on_bound_cylinder(capfd):
    C = Cylinder(center=[0, 0, 1], radius=1, height=2, orientation=[0, 1, 1])
    point = [[-400, 23, 10]]
    C.boundary_normal(point)
    out, _ = capfd.readouterr() 
    assert out == 'Warning: some points are not at the boundary!\n'


def test_output_type_cylinder():
    C = Cylinder(center=[0, 0, 1], radius=1, height=2, orientation=[0, 1, 1])
    i_rand = C.sample_inside(1)
    b_rand = C.sample_boundary(1)
    i_grid = C.sample_inside(1, type='grid')
    b_grid = C.sample_boundary(2, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_bounds_cylinder():
    C = Cylinder(center=[0, 0, 0], radius=3, height=3, orientation=[0, 0, 1])
    bounds = C._compute_bounds()
    assert np.allclose(bounds, [-3, 3, -3, 3, -1.5, 1.5])
    C = Cylinder(center=[0, 0, 0], radius=3, height=3, orientation=[1, 0, 0])
    bounds = C._compute_bounds()
    assert np.allclose(bounds, [-1.5, 1.5, -3, 3, -3, 3])


def test_serialize_cylinder():
    C = Cylinder(center=[0, 0, 0], radius=3, height=3, orientation=[0, 0, 1])
    dct = C.serialize()
    assert dct['dim'] == 3
    assert dct['tol'] == 1e-06
    assert dct['name'] == 'Cylinder'
    assert np.equal(dct['center'], [0, 0, 0]).all()
    assert dct['radius'] == 3
    assert dct['height'] == 3


# Test Polygon3D
def _create_simple_polygon():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, -1]]
    faces = [[0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 2, 4], [0, 1, 4], [1, 2, 4]]
    return vertices, faces


def test_create_with_vertices_faces_poly3D():
    vertices, faces = _create_simple_polygon()
    Polygon3D(vertices=vertices, faces=faces)


def test_throw_error_if_not_data_given_poly3D():
    with pytest.raises(ValueError):
        Polygon3D()


def test_dim_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    assert poly3D.dim == 3


def test_tol_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces, tol=0.1)
    assert poly3D.tol == 0.1


def test_watertight_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    assert poly3D.mesh.is_watertight


def test_surface_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    assert np.isclose(poly3D.surface, 3.6911662)


def test_volume_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    assert np.isclose(poly3D.volume, 1/3.0)


def test_export_and_load_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    poly3D.export_file('testmesh.stl')
    Polygon3D(file_name='testmesh.stl', file_type='stl')
    os.remove('testmesh.stl')


def test_check_inside_and_outside_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = [[0, 0, 0], [0.25, 0.25, 0.25], [0.2, 0.2, -0.1], [0.1, 0.1, 0.78],
              [10, 4, 3], [-0.01, 0, 0]]
    inside = poly3D.is_inside(points)
    assert not inside[0]
    assert all(inside[1:4])
    assert not any(inside[4:6])


def test_check_on_boundary_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = [[0, 0, 0], [0.25, 0.25, 0.25], [0.2, 0.2, -0.1], [0.1, 0.1, 0.78],
              [10, 4, 3], [-0.01, 0, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, -1], 
              [0.5, 0.5, 0]]
    on_bound = poly3D.is_on_boundary(points)
    assert on_bound[0]
    assert not any(on_bound[1:4])
    assert not any(on_bound[4:6])
    assert all(on_bound[6:10])


def test_random_sampling_inside_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = poly3D.sample_inside(545)
    assert np.shape(points) == (545,3)
    assert all(poly3D.is_inside(points))   


def test_grid_sampling_inside_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    # if it is implemented also change the output test |
    #                                                  V
    with pytest.raises(NotImplementedError):
        poly3D.sample_inside(545, type='grid') 


def test_output_type_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    i_rand = poly3D.sample_inside(1)
    b_rand = poly3D.sample_boundary(1)
    #i_grid = poly3D.sample_inside(1, type='grid')
    b_grid = poly3D.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    #assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_random_sampling_boundary_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = poly3D.sample_boundary(515)
    assert np.shape(points) == (515,3)
    assert all(poly3D.is_on_boundary(points)) 


def test_grid_sampling_boundary_poly3D():
    np.random.seed(0)
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = poly3D.sample_boundary(456, type='grid')
    assert np.shape(points) == (456,3)
    assert all(poly3D.is_on_boundary(points)) 


def test_normals_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)    
    points = poly3D.sample_boundary(15, type='grid')
    normals = poly3D.boundary_normal(points)
    assert np.shape(normals) == (15,3)
    for i in range(15):
        assert np.isclose(np.linalg.norm(normals[i]), 1)


def test_normals_direction_poly3D():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    faces = [[0, 1, 3], [0, 2, 3], [0, 1, 2], [1, 2, 3]]
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    points = [[0.2, 0.2, 0], [0, 0.2, 0.2], [0.2, 0, 0.2], [0.4, 0.4, 0.2]]
    normals = poly3D.boundary_normal(points)
    assert np.shape(normals) == (4,3)
    assert np.equal(normals[0], [0, 0, -1]).all()
    assert np.equal(normals[1], [-1, 0, 0]).all()
    assert np.equal(normals[2], [0, -1, 0]).all()
    assert np.allclose(normals[3], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])


def test_console_output_when_not_on_bound_poly3D(capfd):
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)    
    point = [[0.1, 0.1, 10]]
    poly3D.boundary_normal(point)
    out, _ = capfd.readouterr() 
    assert out == 'Warning: some points are not at the boundary!\n'


def test_projection_on_xy_plane():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    # to check if normal vector will be scaled:
    for i in range(1,3):
        p = poly3D.project_on_plane(plane_normal=[0, 0, i], plane_origin=[0, 0, 0])
        bounds = p._compute_bounds()
        assert np.allclose(bounds, [0, 1, 0, 1], atol=1e-06)
        points = [[0, 0], [0.25, 0.25], [0.67, 0.3], [1, 1], [-0.1, 0]]
        inside = p.is_inside(points)
        assert not inside[0]    
        assert not inside[4] 
        assert not inside[3]
        assert inside[1] 
        assert inside[2] 
        assert p.is_on_boundary([[0, 0]])


def test_slice_with_xy_plane():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    p = poly3D.slice_with_plane(plane_normal=[0, 0, 1], plane_origin=[0, 0, 0])
    bounds = p._compute_bounds()
    assert np.allclose(bounds, [0, 1, 0, 1], atol=1e-06)
    points = [[0, 0], [0.25, 0.25], [0.67, 0.3], [1, 1], [-0.1, 0]]
    inside = p.is_inside(points)
    assert not inside[0]    
    assert not inside[4] 
    assert not inside[3]
    assert inside[1] 
    assert inside[2] 
    assert p.is_on_boundary([[0, 0]])


def test_slice_with_plane_parallel_to_xy():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    p = poly3D.slice_with_plane(plane_normal=[0, 0, 1], plane_origin=[0, 0, 0.5])
    bounds = p._compute_bounds()
    assert np.allclose(bounds, [0, 0.5, 0, 0.5], atol=1e-06)
    points = [[0, 0], [0.2, 0.2], [0.67, 0.3], [1, 1], [-0.1, 0]]
    inside = p.is_inside(points)
    assert not inside[0]    
    assert not inside[4] 
    assert not inside[3]
    assert inside[1] 
    assert not inside[2] 
    assert p.is_on_boundary([[0, 0]])


def test_slice_with_plane_empty():
    vertices, faces = _create_simple_polygon()
    poly3D = Polygon3D(vertices=vertices, faces=faces)
    with pytest.raises(ValueError):
        poly3D.slice_with_plane(plane_normal=[0, 0, 2], plane_origin=[0, 0, 10])