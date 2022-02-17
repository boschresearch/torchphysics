import pytest
import torch
import shapely.geometry as s_geo
from shapely.ops import triangulate

from torchphysics.problem.domains.domain2D.shapely_polygon import ShapelyPolygon
from torchphysics.problem.spaces import R2, R1
from torchphysics.problem.spaces.points import Points


# Test ShapelyPolygon
def test_create_poly2D():
    P = ShapelyPolygon(R2('x'), vertices=[[0, 10], [10, 5], [10, 2], [0, 0]])


def test_dim_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [10, 5], [10, 2], [0, 0]])
    assert P.dim == 2


def test_get_volume_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 1], [1, 0], [1, 1]])
    assert P._get_volume() == 0.5


def test_get_volume_poly2D_boundary():
    P = ShapelyPolygon(R2('x'), [[0, 1], [0, 0], [1, 0], [1, 1]])
    assert P.boundary._get_volume() == 4.0


def test_call_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 1], [1, 0], [1, 1]])
    assert P(t=3) == P


def test_call_poly2D_boundary():
    P = ShapelyPolygon(R2('x'), [[0, 1], [1, 0], [1, 1]]).boundary
    assert P(t=3) == P


def test_check_no_input_poly2D():
    with pytest.raises(ValueError):
        _ = ShapelyPolygon(R2('x'))


def test_cant_create_variable_poly2D():
    with pytest.raises(TypeError):
        _ = ShapelyPolygon(R2('x'), vertices=lambda t : t)


def test_ordering_of_corners_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10.0, 5]])
    order = torch.tensor([[0, 10.0], [0, 0], [10, 2], [10, 5], [0, 10]])
    assert torch.equal(torch.tensor(P.polygon.exterior.coords), order)
    P = ShapelyPolygon(R2('x'), [[0, 10.0], [10, 5], [10, 2], [0, 0]])
    assert torch.equal(torch.tensor(P.polygon.exterior.coords), order)


def test_volume_of_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    assert torch.isclose(P._get_volume(), torch.tensor(80.0))


def test_inside_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = torch.tensor([[5, 5], [0, 0], [10, 2.0], [-3, 4]])
    points = Points(points, R2('x'))
    inside = P._contains(points)
    assert inside[0]
    assert not inside[1]
    assert not inside[2]
    assert not inside[3]


def test_on_boundary_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = torch.tensor([[5, 5.0], [0, 0], [10, 2], [-3, 4], [0, 8]])
    points = Points(points, R2('x'))
    on_bound = P.boundary._contains(points)
    assert not on_bound[0]
    assert on_bound[1]
    assert on_bound[2]
    assert not on_bound[3]
    assert on_bound[4]


def test_random_sampling_on_boundary_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]]).boundary
    points = P.sample_random_uniform(n=500)
    assert points.as_tensor.shape == (500, 2)
    assert all(P._contains(points))


def test_grid_sampling_on_boundary_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 0], [10, 10]]).boundary
    points = P.sample_grid(15)
    assert points.as_tensor.shape == (15, 2)
    assert all(P._contains(points))


def test_random_sampling_on_boundary_for_hole_in_poly2D():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = ShapelyPolygon(R2('x'), shapely_polygon=p)
    H = ShapelyPolygon(R2('x'), shapely_polygon=h)
    points = P.boundary.sample_random_uniform(500)
    assert points.as_tensor.shape == (500, 2)
    assert any(H.boundary._contains(points))    
    assert all(P.boundary._contains(points))


def test_grid_sampling_on_boundary_for_hole_in_poly2D():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = ShapelyPolygon(R2('x'), shapely_polygon=p)
    H = ShapelyPolygon(R2('x'), shapely_polygon=h)
    points = P.boundary.sample_grid(500)
    assert points.as_tensor.shape == (500, 2)
    assert any(H.boundary._contains(points))    
    assert all(P.boundary._contains(points))


def test_random_sampling_inside_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    time = Points(torch.tensor([[2.0], [1.1]]), R1('t'))
    points = P.sample_random_uniform(10, params=time)
    assert points.as_tensor.shape == (20, 2)
    assert all(P._contains(points))


def test_random_sampling_inside_poly2D_with_density():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_random_uniform(d=1)
    assert all(P._contains(points))


def test_random_sampling_inside_poly2D_2():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_random_uniform(50)
    assert points.as_tensor.shape == (50, 2)
    assert all(P._contains(points))
    P = ShapelyPolygon(R2('x'), [[0, 0], [0.3, 0], [0.3, 0.9], [0.5, 0.9], [0.5, 0.85], 
                   [1, 0.85], [1, 0.1], [0.4, 0.1], [0.4, 0], [2, 0], 
                   [2, 1], [0, 1]])
    points = P.sample_random_uniform(50)
    assert points.as_tensor.shape == (50, 2)
    assert all(P._contains(points))


def test_add_additional_points_if_some_missing_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 0], [10, 10]])
    T = triangulate(P.polygon)[0]
    points = torch.ones((4, 2))
    n = 4
    points = P._check_enough_points_sampled(n, points, T, 'cpu')
    assert points.shape == (4, 2)
    n = 8
    points = P._check_enough_points_sampled(n, points, T, 'cpu')
    assert points.shape == (8, 2)
    assert all(P._contains(points))


def test_bounds_for_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    bounds = P.bounding_box()
    assert torch.allclose(bounds, torch.tensor([0, 10, 0, 10]).float())
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8], [5, 20]])
    bounds = P.bounding_box()
    assert torch.allclose(bounds, torch.tensor([0, 10, 0, 20]).float())


def test_grid_sampling_inside_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_grid(250)
    assert points.as_tensor.shape == (250, 2)
    assert all(P._contains(points))


def test_grid_sampling_inside_poly2D_no_extra_points():
    P = ShapelyPolygon(R2('x'), [[0, 1], [1, 1], [1, 0], [0, 0]])
    points = P.sample_grid(100)
    assert points.as_tensor.shape == (100, 2)
    assert all(P._contains(points))


def test_grid_sampling_inside_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_grid(d=11)
    assert all(P._contains(points))


def test_random_sampling_inside_concav_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 0], [0, -5], [-10, -5], [-10, -10], [10, -10],
                   [10, 10], [-10, 10], [-10, 0]])
    points = P.sample_grid(263)
    assert points.as_tensor.shape == (263, 2)
    assert all(P._contains(points))


def test_boundary_normal_for_concav_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 0], [0, -10], [10, -10], [10, 10], [-10, 10], [-10, 0]])
    points = torch.tensor([[0, -5], [5, -10], [5, 10], [-10, 7], [-4, 0], [10, 0]])
    points = Points(points, R2('x'))
    normals = P.boundary.normal(points)
    assert torch.allclose(normals[0], torch.tensor([-1.0, 0]))
    assert torch.allclose(normals[1], torch.tensor([0.0, -1]))
    assert torch.allclose(normals[2], torch.tensor([0.0, 1]))
    assert torch.allclose(normals[3], torch.tensor([-1.0, 0]))
    assert torch.allclose(normals[4], torch.tensor([0.0, -1]))
    assert torch.allclose(normals[5], torch.tensor([1.0, 0]))


def test_boundary_normal_poly2D():
    P = ShapelyPolygon(R2('x'), [[0, 10], [0, 0], [10, 2], [10, 8]])
    points = torch.tensor([[0, 5], [10, 5], [1, 2.0/10], [9, 8+2.0/10]])
    points = Points(points, R2('x'))
    normals = P.boundary.normal(points)
    assert torch.allclose(normals[0], torch.tensor([-1.0, 0]))
    assert torch.allclose(normals[1], torch.tensor([1.0, 0]))
    norm = torch.sqrt(torch.tensor(2**2+10**2))
    assert torch.allclose(normals[3], torch.tensor([2.0/norm, 10/norm]))
    assert torch.allclose(normals[2], torch.tensor([2.0/norm, -10/norm]))


def test_boundary_normal_poly2D_with_hole():
    h = s_geo.Polygon(shell=[[0.15, 0.15], [0.25, 0.15], [0.15, 0.25]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = ShapelyPolygon(R2('x'), shapely_polygon=p)
    points = torch.tensor([[0.5, 0], [0, 0.5], [0.2, 0.15]])
    points = Points(points, R2('x'))
    normals = P.boundary.normal(points)
    assert normals.shape == (3, 2)
    assert torch.allclose(normals[0], torch.tensor([0.0, -1]))
    assert torch.allclose(normals[1], torch.tensor([-1.0, 0]))
    assert torch.allclose(normals[2], torch.tensor([0.0, 1]))