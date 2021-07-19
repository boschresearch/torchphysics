import pytest
import numpy as np
import os

from torchphysics.problem.domain.domain3D import Polygon3D


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