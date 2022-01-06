import pytest
import numpy as np
import torch
import os

from torchphysics.problem.spaces.space import R3, R2, R1
from torchphysics.problem.domains.domain3D.trimesh_polyhedron import \
    TrimeshPolyhedron, TrimeshBoundary
from torchphysics.problem.spaces.points import Points


def _create_simple_polygon():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, -1]]
    faces = [[0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 2, 4], [0, 1, 4], [1, 2, 4]]
    return vertices, faces


def test_create_with_vertices_faces_poly3D():
    vertices, faces = _create_simple_polygon()
    TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)


def test_throw_error_if_callable_vetrices_given_poly3D():
    with pytest.raises(TypeError):
        TrimeshPolyhedron(R3('x'), vertices=lambda t:t, faces=[[1, 2, 3]])


def test_throw_error_if_not_data_given_poly3D():
    with pytest.raises(ValueError):
        TrimeshPolyhedron(R3('x'))


def test_dim_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    assert poly3D.dim == 3


def test_call_poly3D():
    vertices, faces = _create_simple_polygon()
    p = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    assert p(t=0) == p


def test_bounding_box_poly3D():
    vertices, faces = _create_simple_polygon()
    p = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    bounds = p.bounding_box()
    assert bounds[0] == 0 
    assert bounds[1] == 1 
    assert bounds[2] == 0 
    assert bounds[3] == 1 
    assert bounds[4] == -1 
    assert bounds[5] == 1 


def test_watertight_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    assert poly3D.mesh.is_watertight


def test_volume_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    assert np.isclose(poly3D._get_volume(), 1/3.0)


def test_export_and_load_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    poly3D.export_file('testmesh.stl')
    TrimeshPolyhedron(R3('x'), file_name='testmesh.stl', file_type='stl')
    os.remove('testmesh.stl')


def test_check_inside_and_outside_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = [[0, 0, 0], [0.25, 0.25, 0.25], [0.2, 0.2, -0.1], [0.1, 0.1, 0.78],
              [10, 4, 3], [-0.01, 0, 0]]
    points = Points(torch.tensor(points), R3('x'))
    inside = poly3D._contains(points)
    assert not inside[0]
    assert all(inside[1:4])
    assert not any(inside[4:6])


def test_check_on_boundary_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = [[0, 0, 0], [0.25, 0.25, 0.25], [0.2, 0.2, -0.1], [0.1, 0.1, 0.78],
              [10, 4, 3], [-0.01, 0, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, -1], 
              [0.5, 0.5, 0]]
    points = Points(torch.tensor(points), R3('x'))
    on_bound = poly3D.boundary._contains(points)
    assert on_bound[0]
    assert not any(on_bound[1:4])
    assert not any(on_bound[4:6])
    assert all(on_bound[6:10])


def test_random_sampling_inside_poly3D_with_n():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.sample_random_uniform(n=545)
    assert points.as_tensor.shape == (545,3)
    assert all(poly3D._contains(points))   


def test_random_sampling_inside_poly3D_with_d():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.sample_random_uniform(d=13)
    assert all(poly3D._contains(points)) 


def test_random_sampling_inside_poly3D_with_n_and_params():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    time = Points(torch.tensor([[1.2], [9]]), R1('t'))
    points = poly3D.sample_random_uniform(n=20, params=time)
    assert points.as_tensor.shape == (40,3)
    assert all(poly3D._contains(points))   


def test_grid_sampling_inside_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.sample_grid(n=20)
    assert points.as_tensor.shape == (20, 3)
    assert all(torch.logical_or(poly3D._contains(points),
                                poly3D.boundary._contains(points)))    


def test_grid_sampling_inside_poly3D_with_d():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.sample_grid(d=13)
    assert all(torch.logical_or(poly3D._contains(points),
                                poly3D.boundary._contains(points)))   


def test_grid_sampling_inside_poly3D_with_arbritray_params():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    time = Points(torch.tensor([[1.0]]), R1('t'))
    points = poly3D.sample_grid(n=50, params=time)
    assert points.as_tensor.shape == (50, 3)
    assert all(torch.logical_or(poly3D._contains(points),
                                poly3D.boundary._contains(points)))   


def test_random_sampling_boundary_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.boundary.sample_random_uniform(n=50)
    assert points.as_tensor.shape == (50,3)
    assert all(poly3D.boundary._contains(points)) 


def test_random_sampling_boundary_poly3D_with_d():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.boundary.sample_random_uniform(d=11)
    assert all(poly3D.boundary._contains(points)) 


def test_random_sampling_boundary_poly3D_with_n_and_params():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    time = Points(torch.tensor([[1.2], [9]]), R1('t'))
    points = poly3D.boundary.sample_random_uniform(n=50, params=time)
    assert points.as_tensor.shape == (100,3)
    assert all(poly3D.boundary._contains(points)) 


def test_grid_sampling_boundary_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.boundary.sample_grid(50)
    assert points.as_tensor.shape == (50,3)
    assert all(poly3D.boundary._contains(points)) 


def test_grid_sampling_boundary_poly3D_with_d():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    points = poly3D.boundary.sample_grid(d=12)
    assert all(poly3D.boundary._contains(points)) 


def test_volume_poly3D_boundary():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces).boundary
    assert np.isclose(poly3D._get_volume().item(), 3.69, atol=0.02)


def test_normals_poly3D():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)    
    points = poly3D.boundary.sample_grid(n=15)
    normals = poly3D.boundary.normal(points)
    assert normals.shape == (15,3)
    for i in range(15):
        assert torch.isclose(torch.linalg.norm(normals[i]), torch.tensor(1.0))


def test_normals_direction_poly3D():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    faces = [[0, 1, 3], [0, 2, 3], [0, 1, 2.0], [1, 2, 3]]
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces).boundary
    points = [[0.2, 0.2, 0], [0, 0.2, 0.2], [0.2, 0, 0.2], [0.4, 0.4, 0.2]]
    points = Points(torch.tensor(points), R3('x'))
    normals = poly3D.normal(points)
    assert normals.shape == (4, 3)
    assert torch.allclose(normals[0], torch.tensor([0.0, 0, -1.0]))
    assert torch.allclose(normals[1], torch.tensor([-1.0, 0, 0]))
    assert torch.allclose(normals[2], torch.tensor([0.0, -1.0, 0]))
    sqrt_3 = 1.0/torch.sqrt(torch.tensor(3.0))
    assert torch.allclose(normals[3], sqrt_3 * torch.tensor([1.0, 1.0, 1.0]))


def test_projection_on_xy_plane():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    # to check if normal vector will be scaled:
    for i in range(1,3):
        p = poly3D.project_on_plane(new_space=R2('x'), 
                                    plane_normal=[0, 0, i], plane_origin=[0, 0, 0])
        bounds = p.bounding_box()
        assert np.allclose(bounds, [0, 1, 0, 1], atol=1e-06)
        points = [[0, 0], [0.25, 0.25], [0.67, 0.3], [1, 1], [-0.1, 0]]
        points = Points(torch.tensor(points), R2('x'))
        inside = p._contains(points)
        assert not inside[0]    
        assert not inside[4] 
        assert not inside[3]
        assert inside[1] 
        assert inside[2] 
        assert p.boundary._contains(Points(torch.tensor([[0, 0.0]]), R2('x')))


def test_slice_with_xy_plane():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    p = poly3D.slice_with_plane(new_space=R2('x'), plane_normal=[0, 0, 1],
                                plane_origin=[0, 0, 0])
    bounds = p.bounding_box()
    assert np.allclose(bounds, [0, 1, 0, 1], atol=1e-06)
    points = [[0, 0], [0.25, 0.25], [0.67, 0.3], [1, 1], [-0.1, 0]]
    points = Points(torch.tensor(points), R2('x'))
    inside = p._contains(points)
    assert not inside[0]    
    assert not inside[4] 
    assert not inside[3]
    assert inside[1] 
    assert inside[2] 
    assert p.boundary._contains(Points(torch.tensor([[0, 0.0]]), R2('x')))


def test_slice_with_plane_parallel_to_xy():
    vertices, faces = _create_simple_polygon()
    poly3D = TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    p = poly3D.slice_with_plane(new_space=R2('x'), 
                                plane_normal=[0, 0, 1], plane_origin=[0, 0, 0.5])
    bounds = p.bounding_box()
    assert np.allclose(bounds, [0, 0.5, 0, 0.5], atol=1e-06)
    points = [[0, 0], [0.2, 0.2], [0.67, 0.3], [1, 1], [-0.1, 0]]
    points = Points(torch.tensor(points), R2('x'))
    inside = p._contains(points)
    assert not inside[0]    
    assert not inside[4] 
    assert not inside[3]
    assert inside[1] 
    assert not inside[2] 
    assert p.boundary._contains(Points(torch.tensor([[0, 0.0]]), R2('x')))


def test_slice_with_plane_empty():
    vertices, faces = _create_simple_polygon()
    poly3D =TrimeshPolyhedron(R3('x'), vertices=vertices, faces=faces)
    with pytest.raises(RuntimeError):
        poly3D.slice_with_plane(R2('x'), plane_normal=[0, 0, 2],
                                plane_origin=[0, 0, 10])