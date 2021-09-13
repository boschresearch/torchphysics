import pytest
import numpy as np
import matplotlib.patches as patches
import shapely.geometry as s_geo
from shapely.ops import triangulate
from torchphysics.problem.domain.domain2D import (Rectangle, Circle,
                                                  Triangle, Polygon2D)

# Tests for rectangle


def test_create_rectangle():
    _ = Rectangle([0, 0], [1, 0], [0, 1])


def test_throw_error_if_not_rectangle():
    with pytest.raises(ValueError):
        _ = Rectangle([0, 0], [2, 0], [2, 0])


def test_check_dim_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    assert R.dim == 2


def test_volume_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    assert np.isclose(R.volume, 1)


def test_surface_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    assert np.isclose(R.surface, 4)


def test_check_corners_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    assert all((R.corner_dl == [0, 0]))
    assert all((R.corner_dr == [1, 0]))
    assert all((R.corner_tl == [0, 1]))


def test_check_side_length_rect():
    R = Rectangle([0, 0], [1, 0], [0, 2])
    assert R.length_lr == 1
    assert R.length_td == 2


def test_check_inverse_matrix_rect():
    R = Rectangle([0, 0], [1, 0], [0, 2])
    assert R.inverse_matrix[0][0] == 1
    assert R.inverse_matrix[0][1] == 0
    assert R.inverse_matrix[1][0] == 0
    assert R.inverse_matrix[1][1] == 1/2


def test_output_type_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    i_rand = R.sample_inside(1)
    b_rand = R.sample_boundary(1)
    i_grid = R.sample_inside(1, type='grid')
    b_grid = R.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_check_points_inside_and_outside_rect():
    R = Rectangle([1, 0], [2, 1], [2, -1])
    points = [[2, 0], [0, 0]]
    inside = R.is_inside(points)
    assert inside[0]
    assert not inside[1]


def test_check_points_boundary_rect():
    R = Rectangle([0, 0], [1, 0], [0, 3])
    points = [[0, 0.2], [1, 0], [0.5, 0.5], [-2, 10], [0, -0.1]]
    bound = R.is_on_boundary(points)
    assert all(bound[0:1])
    assert not all(bound[2:])


def test_random_sampling_inside_rect():
    np.random.seed(0)
    R = Rectangle([0, 0], [1, 1], [-2, 2])
    points = R.sample_inside(10, type='random')
    compare = [[-1.0346366, 2.1322637], [-0.34260046, 1.7729793],
               [-0.53332573, 1.7388525], [-1.30631, 2.3960764],
               [0.28158268, 0.56572694], [0.47163552, 0.8201527],
               [0.39715043, 0.478024], [-0.7734667, 2.5570128],
               [-0.5926508, 2.5199764], [-1.3565828, 2.1234658]]
    compare = np.array(compare).astype(np.float32)
    assert np.shape(points) == (10, 2)
    assert all(R.is_inside(points))
    assert all((compare == points).all(axis=1))


def test_grid_sampling_inside_rect():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    points = R.sample_inside(250, type='grid')
    assert np.shape(points) == (250, 2)
    assert all(R.is_inside(points))
    assert np.linalg.norm(points[0]-points[1]) == np.linalg.norm(points[2]-points[1])


def test_normal_sampling_inside_rect():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    v_matrix = [[1, 0], [0, 2]]
    points = R.sample_inside(250, type='normal', sample_params={'mean': [0, 0],
                                                                'cov': v_matrix})
    assert np.shape(points) == (250, 2)
    assert all(R.is_inside(points))


def test_lhs_sampling_inside_rect():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    points = R.sample_inside(50, type='lhs')
    assert np.shape(points) == (50, 2)
    assert all(R.is_inside(points))


def test_random_sampling_boundary_rect():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    points = R.sample_boundary(10, type='random')
    assert np.shape(points) == (10, 2)
    assert all(R.is_on_boundary(points))


def test_grid_sampling_boundary_rect():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    points = R.sample_boundary(10, type='grid')
    assert np.shape(points) == (10, 2)
    assert all(R.is_on_boundary(points))
    assert np.linalg.norm(points[0]-points[1]) == np.linalg.norm(points[2]-points[1])


def test_normal_sampling_boundary_rect():
    R = Rectangle([0, 0], [2, 0], [0, 2])
    points = R.sample_boundary(10, type='normal', sample_params={'mean': [0, 0], 
                                                                 'cov': 0.1})
    assert np.shape(points) == (10, 2)
    assert all(R.is_on_boundary(points))
    points = R.sample_boundary(10, type='normal', sample_params={'mean': [0, 0.05], 
                                                                 'cov': 0.1})
    assert np.shape(points) == (10, 2)
    assert all(R.is_on_boundary(points))


def test_find_position_on_boundary_when_point_not_on_boundary():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    with pytest.raises(ValueError):
        _ = R._find_position_on_boundary([-1, -1], np.array([[0, 0]]), [2])
    with pytest.raises(ValueError):
        _ = R._normal_sampling_boundary(20, mean=[-1, -1], cov=3)


def test_grid_sampling_boundary_rect_for_side_lengths_zero():
    R = Rectangle([0, 0], [2, 1], [-1, 2])
    R.length_lr = 0
    R.length_td = 0
    points = R.sample_boundary(10, type='grid')
    assert np.shape(points) == (0, 2)


def test_boundary_normal_rect():
    R = Rectangle([0, 0], [2, 0], [0, 2])
    point = [[0, 0.5], [0.7, 2]]
    normal = R.boundary_normal(point)
    assert normal[0][0] == -1
    assert normal[0][1] == 0
    assert normal[1][0] == 0
    assert normal[1][1] == 1


def test_normal_vector_at_corner_():
    R = Rectangle([0, 0], [2, 0], [0, 2])
    point = [[0, 0], [2, 0], [0, 2], [2, 2]]
    normal = R.boundary_normal(point)
    for i in range(4):
        assert np.isclose(np.linalg.norm(normal[i]), 1)

    sqrt_2 = np.sqrt(2)
    assert normal[0][0] == -1/sqrt_2
    assert normal[0][1] == -1/sqrt_2
    assert normal[1][0] == 1/sqrt_2
    assert normal[1][1] == -1/sqrt_2
    assert normal[2][0] == -1/sqrt_2
    assert normal[2][1] == 1/sqrt_2
    assert normal[3][0] == 1/sqrt_2
    assert normal[3][1] == 1/sqrt_2


def test_normal_vector_for_rotated_rect():
    R = Rectangle([1, 0], [2, -1], [2, 1])
    point = [[1.5, -0.5], [1.5, 0.5], [1.5, 1.5]]
    normal = R.boundary_normal(point)
    assert np.isclose(np.dot([1, -1], normal[0]), 0)
    assert np.isclose(np.dot([1, 1], normal[1]), 0)
    assert np.isclose(np.dot([1, -1], normal[2]), 0)


def test_console_output_when_not_on_bound_rect(capfd):
    R = Rectangle([0, 0], [9, 0], [0, 5])
    point = [[3, 3]]
    R.boundary_normal(point)
    out, err = capfd.readouterr()
    assert out == 'Warning: some points are not at the boundary!\n'


def test_grid_for_plot_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    points = R.grid_for_plots(150)
    inside = R.is_inside(points)
    bound = R.is_on_boundary(points)
    assert all(np.logical_or(inside, bound))


def test_bounds_for_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    bounds = R._compute_bounds()
    assert bounds == [0, 1, 0, 1]
    R = Rectangle([0, 0], [1, -1], [1, 1])
    bounds = R._compute_bounds()
    assert bounds == [0, 2, -1, 1]


def test_outline_rect():
    R = Rectangle([0, 0], [2, 0], [0, 1])
    outline = R.outline()
    assert isinstance(outline, patches.Rectangle)
    assert outline.xy == (0,0)
    assert outline.get_height() == 1
    assert outline.get_width() == 2
    assert outline.angle == 360.0


def test_serialize_rect():
    R = Rectangle([0, 0], [1, 0], [0, 1])
    dct = R.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == 'Rectangle'
    assert np.equal(dct['corner_dl'], [0, 0]).all()
    assert np.equal(dct['corner_tl'], [0, 1]).all()
    assert np.equal(dct['corner_dr'], [1, 0]).all()


# Test for circle
def test_create_circle():
    _ = Circle([0, 0], 2)


def test_check_dim_circle():
    C = Circle([0, 0], 2)
    assert C.dim == 2


def test_check_center_and_radius():
    C = Circle([0.5, 0.6], 2)
    assert C.center[0] == 0.5
    assert C.center[1] == 0.6
    assert C.radius == 2


def test_volume_circle():
    C = Circle([0, 0], 2)
    assert np.isclose(C.volume, np.pi*4)


def test_surface_circle():
    C = Circle([0.5, 0.6], 2)
    assert np.isclose(C.surface, 2*np.pi*2)


def test_output_type_circle():
    C = Circle([1, 0], 3)
    i_rand = C.sample_inside(1)
    b_rand = C.sample_boundary(1)
    i_grid = C.sample_inside(1, type='grid')
    b_grid = C.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_check_points_inside_and_outside_circle():
    C = Circle([1, 0], 2)
    points = [[2, 0], [1, 1], [10, 0]]
    inside = C.is_inside(points)
    assert all(inside[0:2])
    assert not inside[2]


def test_check_points_boundary_circle():
    C = Circle([1, 0], 2)
    points = [[2, 0], [1, 1], [3, 0], [1, 2]]
    inside = C.is_on_boundary(points)
    assert not all(inside[0:2])
    assert all(inside[2:])


def test_random_sampling_inside_circle():
    np.random.seed(0)
    C = Circle([1, 1], 3)
    points = C.sample_inside(5, type='random')
    compare = [[-0.35227352, -0.7637114], [-1.344475,  1.9696087],
               [2.8110569, -0.46456257], [3.157019,  0.4987838],
               [-0.4519993,  2.3056]]
    compare = np.array(compare).astype(np.float32)
    assert np.shape(points) == (5, 2)
    assert all(C.is_inside(points))
    assert all((compare == points).all(axis=1))


def test_grid_sampling_inside_circle():
    C = Circle([0, 0], 4)
    points = C.sample_inside(258, type='grid')
    assert np.shape(points) == (258, 2)
    assert all(C.is_inside(points))


def test_lhs_sampling_inside_circle():
    C = Circle([0, 0], 4)
    points = C.sample_inside(258, type='lhs')
    assert np.shape(points) == (258, 2)
    assert all(C.is_inside(points))


def test_random_sampling_boundary_circle():
    np.random.seed(0)
    C = Circle([0, 0], 4.5)
    points = C.sample_boundary(10, type='random')
    compare = [[-4.290002, -1.3586327], [-0.9764186, -4.3927903],
               [-3.5941048, -2.7078424], [-4.322242, -1.2522879],
               [-3.992119,  2.076773], [-2.7380629, -3.571136],
               [-4.158401,  1.719797], [3.4990482, -2.8296046],
               [4.3832226, -1.0185084], [-3.3461978,  3.0088139]]
    compare = np.array(compare).astype(np.float32)
    assert np.shape(points) == (10, 2)
    assert all(C.is_on_boundary(points))
    assert all((compare == points).all(axis=1))


def test_grid_sampling_boundary_circle():
    C = Circle([0, 0], 2)
    points = C.sample_boundary(150, type='grid')
    assert np.shape(points) == (150, 2)
    assert all(C.is_on_boundary(points))
    for i in range(len(points)-2):
        assert np.isclose(np.linalg.norm(points[i]-points[i+1]),
                          np.linalg.norm(points[i+2]-points[i+1]))


def test_normal_sampling_boundary_circle():
    C = Circle([0, 0], 2)
    points = C.sample_boundary(150, type='normal', sample_params={'mean': [2, 0], 
                                                                  'cov': 2})
    assert np.shape(points) == (150, 2)
    assert all(C.is_on_boundary(points))
    points = C.sample_boundary(150, type='normal', sample_params={'mean': [0, -2], 
                                                                  'cov': 2})
    assert np.shape(points) == (150, 2)
    assert all(C.is_on_boundary(points))


def test_boundary_normal_circle():
    C = Circle([1, 0], 2)
    point = [[3, 0], [1, 2]]
    normal = C.boundary_normal(point)
    assert normal[0][0] == 1
    assert normal[0][1] == 0
    assert normal[1][0] == 0
    assert normal[1][1] == 1


def test_normal_vector_length_circle():
    C = Circle([0, 0], 5)
    points = C.sample_boundary(50)
    normal = C.boundary_normal(points)
    for i in range(len(points)):
        assert np.isclose(np.linalg.norm(normal[i]), 1)


def test_console_output_when_not_on_bound_circle(capfd):
    C = Circle([0, 0], 5)
    point = [[0, 0]]
    C.boundary_normal(point)
    out, err = capfd.readouterr()
    assert out == 'Warning: some points are not at the boundary!\n'


def test_grid_for_plot_circle():
    C = Circle([0, 0], 5)
    points = C.grid_for_plots(150)
    inside = C.is_inside(points)
    bound = C.is_on_boundary(points)
    assert all(np.logical_or(inside, bound))


def test_bounds_for_circle():
    C = Circle([1, 0], 5)
    bounds = C._compute_bounds()
    assert bounds == [-4, 6, -5, 5]


def test_outline_circle():
    C = Circle([1,0], 3)
    outline = C.outline()
    assert isinstance(outline, patches.Circle)
    assert outline.get_radius() == 3


def test_serialize_circle():
    C = Circle([1, 0], 5)
    dct = C.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == 'Circle'
    assert np.equal(dct['center'], [1, 0]).all()
    assert np.equal(dct['radius'], 5).all()


# Test for triangle class
def test_create_triangle():
    Triangle([0, 0], [1, 0], [0, 1])


def test_check_dim_triangle():
    T = Triangle([0, 0], [1, 0], [0, 1])
    assert T.dim == 2


def test_tol_tirangle():
    T = Triangle([0, 0], [1, 0], [0, 1], tol=0.5)
    assert T.tol == 0.5


def test_area_triangle():
    T = Triangle([0, 0], [1, 0], [0, 1])
    assert T.volume == 1/2


def test_reorder_corners_triangle():
    T = Triangle([0, 0], [0, 1], [1, 0])
    assert T.volume == 1/2
    assert all(np.equal(T.corners[1], [1, 0]))
    assert all(np.equal(T.corners[2], [0, 1]))
    assert all(np.equal(T.corners[0], T.corners[3]))


def test_surface_triangle():
    T = Triangle([0, 0], [0, 4], [4, 0])
    assert np.isclose(T.surface, 8+np.sqrt(32))


def test_side_lengths_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    assert np.isclose(T.side_lengths[0], 2)
    assert np.isclose(T.side_lengths[1], np.sqrt(5))
    assert np.isclose(T.side_lengths[2], 1)


def test_normals_triangle():
    T = Triangle([0, 0], [1, 0], [0, 1])
    assert len(T.normals) == 3
    assert all(np.equal(T.normals[0], [0, -1]))
    assert np.isclose(T.normals[1][0], 1/np.sqrt(2))
    assert np.isclose(T.normals[1][1], 1/np.sqrt(2))
    assert all(np.equal(T.normals[2], [-1, 0]))
    for i in range(3):
        assert np.isclose(np.dot(T.normals[i], T.corners[i+1]-T.corners[i]), 0)
    for i in range(3):
        assert np.isclose(np.linalg.norm(T.normals[i]), 1)


def test_points_inside_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    points = [[-0.1, 0.1], [0.25, 0.25], [1, 1.5], [1.5, -0.1]]
    inside = T.is_inside(points).flatten()
    assert not inside[0]
    assert inside[1]
    assert not inside[2]
    assert not inside[3]


def test_output_type_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    i_rand = T.sample_inside(1)
    b_rand = T.sample_boundary(1)
    i_grid = T.sample_inside(1, type='grid')
    b_grid = T.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_random_sampling_inside_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    points = T.sample_inside(500)
    assert np.shape(points) == (500, 2)
    assert all(T.is_inside(points))


def test_bounds_for_triangle():
    T = Triangle([0, 0], [1, 0], [0, 1])
    bounds = T._compute_bounds()
    assert bounds == [0, 1, 0, 1]
    T = Triangle([-5, 0], [1, 1], [0, 2])
    bounds = T._compute_bounds()
    assert bounds == [-5, 1, 0, 2]


def test_grid_sampling_inside_triangle():
    T = Triangle([0, 0], [1, 5], [-2, 2])
    points = T.sample_inside(521, type='grid')
    assert np.shape(points) == (521, 2)
    assert all(T.is_inside(points))


def test_grid_sampling_inside_pointed_triangle():
    T = Triangle([0, 0], [10, 5], [10, 6])
    points = T.sample_inside(521, type='grid')
    assert np.shape(points) == (521, 2)
    assert all(T.is_inside(points))


def test_lhs_sampling_inside_triangle():
    T = Triangle([0, 0], [1, 5], [-2, 2])
    points = T.sample_inside(21, type='lhs')
    assert np.shape(points) == (21, 2)
    assert all(T.is_inside(points))


def test_points_on_boundary_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    points = [[-0.1, 0.1], [0.25, 0.25], [2, 0], [0, 0], [0, 0.5], [0, -0.5]]
    on_bound = T.is_on_boundary(points).flatten()
    assert not on_bound[0]
    assert not on_bound[1]
    assert on_bound[2]
    assert on_bound[3]
    assert on_bound[4]
    assert not on_bound[5]


def test_bondary_normal_triangle():
    T = Triangle([0, 0], [2, 0], [0, 1])
    points = [[0.25, 0], [0, 0.5], [1.5, 0.25]]
    normals = T.boundary_normal(points)
    assert len(normals) == 3
    assert all(np.equal(normals[0], [0, -1]))
    assert all(np.equal(normals[1], [-1, 0]))
    assert np.isclose(normals[2][0], 1/np.sqrt(5))
    assert np.isclose(normals[2][1], 2/np.sqrt(5))


def test_console_output_when_not_on_bound_triangle(capfd):
    T = Triangle([0, 0], [2, 0], [0, 1])
    point = [[0.1, 0.1]]
    T.boundary_normal(point)
    out, _ = capfd.readouterr()
    assert out == 'Warning: some points are not at the boundary!\n'


def test_random_sampling_boundary_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    points = T.sample_boundary(500)
    assert np.shape(points) == (500, 2)
    assert all(T.is_on_boundary(points))


def test_grid_sampling_boundary_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    points = T.sample_boundary(500, type='grid')
    assert np.shape(points) == (500, 2)
    assert all(T.is_on_boundary(points))


def test_normal_sampling_boundary_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    points = T.sample_boundary(50, type='normal', sample_params={'mean':[0, 10], 
                                                                 'cov': 1})
    assert np.shape(points) == (50, 2)
    assert all(T.is_on_boundary(points))


def test_grid_sampling_boundary_triangle_for_side_lengths_zero():
    T = Triangle([0, 0], [2, 0], [0, 1])
    for i in range(len(T.side_lengths)):
        T.side_lengths[i] = 0
    points = T.sample_boundary(10, type='grid')
    assert np.shape(points) == (0, 2)


def test_grid_for_plot_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    points = T.grid_for_plots(150)
    inside = T.is_inside(points)
    bound = T.is_on_boundary(points)
    assert all(np.logical_or(inside, bound))


def test_outline_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    outline = T.outline()
    assert isinstance(outline, patches.Polygon)
    edges = outline.get_xy()
    assert np.allclose(edges, [[0, 10], [-12, 2], [13, 5], [0, 10]])


def test_serialize_triangle():
    T = Triangle([0, 10], [13, 5], [-12, 2])
    dct = T.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 1e-06
    assert dct['name'] == 'Triangle'
    assert np.equal(dct['corner_1'], [0, 10]).all()
    assert np.equal(dct['corner_3'], [13, 5]).all()
    assert np.equal(dct['corner_2'], [-12, 2]).all()


# Test Polygon2D
def test_create_poly2D():
    P = Polygon2D([[0, 10], [10, 5], [10, 2], [0, 0]])


def test_dim_poly2D():
    P = Polygon2D([[0, 10], [10, 5], [10, 2], [0, 0]])
    assert P.dim == 2


def test_tol_poly2D():
    P = Polygon2D([[0, 10], [10, 5], [10, 2], [0, 0]], tol=2)
    assert P.tol == 2


def test_check_triangle_poly2D():
    with pytest.raises(ValueError):
        _ = Polygon2D([[0, 10], [10, 5], [10, 2]])


def test_check_no_input_poly2D():
    with pytest.raises(ValueError):
        _ = Polygon2D()


def test_ordering_of_corners_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 5]])
    order = [[0, 10], [0, 0], [10, 2], [10, 5], [0, 10]]
    assert np.equal(P.polygon.exterior.coords, order).all()
    P = Polygon2D([[0, 10], [10, 5], [10, 2], [0, 0]])
    assert np.equal(P.polygon.exterior.coords, order).all()


def test_surface_of_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    assert np.isclose(P.surface, 16+2*np.sqrt(2**2+10**2))


def test_volume_of_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    assert np.isclose(P.volume, 80)


def test_normals_of_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    assert len(P.exterior_normals) == 4
    assert np.equal(P.exterior_normals[0], [-1, 0]).all()
    assert np.equal(P.exterior_normals[2], [1, 0]).all()
    norm = np.sqrt(2**2+10**2)
    assert np.allclose(P.exterior_normals[1], [2/norm, -10/norm])
    assert np.allclose(P.exterior_normals[3], [2/norm, 10/norm])
    assert len(P.inner_normals) == 0


def test_normals_of_poly2D_with_hole():
    h = s_geo.Polygon(shell=[[1, 1], [3, 1], [1, 3]])
    p = s_geo.Polygon(shell=[[0, 10], [0, 0], [10, 2], [10, 8]],
                      holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    assert len(P.exterior_normals) == 4
    assert np.equal(P.exterior_normals[0], [-1, 0]).all()
    assert np.equal(P.exterior_normals[2], [1, 0]).all()
    norm = np.sqrt(2**2+10**2)
    assert np.allclose(P.exterior_normals[1], [2/norm, -10/norm])
    assert np.allclose(P.exterior_normals[3], [2/norm, 10/norm])
    assert len(P.inner_normals) == 1
    assert len(P.inner_normals[0]) == 3
    assert np.equal(P.inner_normals[0][2], [0, 1]).all()
    assert np.allclose(P.inner_normals[0][1], [-1/np.sqrt(2), -1/np.sqrt(2)])
    assert np.equal(P.inner_normals[0][0], [1, 0]).all()


def test_inside_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = [[5, 5], [0, 0], [10, 2], [-3, 4]]
    inside = P.is_inside(points)
    assert inside[0]
    assert not inside[1]
    assert not inside[2]
    assert not inside[3]


def test_on_boundary_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = [[5, 5], [0, 0], [10, 2], [-3, 4], [0, 8]]
    on_bound = P.is_on_boundary(points)
    assert not on_bound[0]
    assert on_bound[1]
    assert on_bound[2]
    assert not on_bound[3]
    assert on_bound[4]


def test_grid_for_plot_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.grid_for_plots(150)
    inside = P.is_inside(points)
    bound = P.is_on_boundary(points)
    assert all(np.logical_or(inside, bound))


def test_random_sampling_on_boundary_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_boundary(500)
    assert np.shape(points) == (500, 2)
    assert all(P.is_on_boundary(points))


def test_grid_sampling_on_boundary_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_boundary(15, type='grid')
    assert np.shape(points) == (15, 2)
    assert all(P.is_on_boundary(points))


def test_normal_sampling_on_boundary_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_boundary(15, type='normal', sample_params={'mean': [10, 0], 
                                                                 'cov': 0.2})
    assert np.shape(points) == (15, 2)
    assert all(P.is_on_boundary(points))


def test_normal_sampling_on_boundary_poly2D_point_not_on_bound():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    with pytest.raises(ValueError):
        _ = P.sample_boundary(15, type='normal', sample_params={'mean': [100, 0], 
                                                                'cov': 0.2})


def test_normal_sampling_on_boundary_for_hole_in_poly2D():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    points = P.sample_boundary(50, type='normal', sample_params={'mean': [0.20, 0.15], 
                                                                 'cov': 0.01})
    assert np.shape(points) == (50, 2)
    assert all(P.is_on_boundary(points))


def test_normal_sampling_inside_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_inside(15, type='normal', sample_params={'mean': [10, 0], 
                                                               'cov': 0.2})
    assert np.shape(points) == (15, 2)
    assert all(P.is_inside(points))



def test_random_sampling_on_boundary_for_hole_in_poly2D():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    H = Polygon2D(shapely_polygon=h)
    points = P.sample_boundary(500)
    assert np.shape(points) == (500, 2)
    assert any(H.is_on_boundary(points))    
    assert all(P.is_on_boundary(points))


def test_grid_sampling_on_boundary_for_hole_in_poly2Dpoly2D():
    h = s_geo.Polygon(shell=[[0.20, 0.15], [0.5, 0.25], [0.25, 0.5]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    H = Polygon2D(shapely_polygon=h)
    points = P.sample_boundary(50, type='grid')
    assert np.shape(points) == (50, 2)
    assert any(H.is_on_boundary(points))    
    assert all(P.is_on_boundary(points))


def test_random_sampling_inside_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_inside(583)
    assert np.shape(points) == (583, 2)
    assert all(P.is_inside(points))


def test_random_sampling_inside_poly2D_2():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_inside(50)
    assert np.shape(points) == (50, 2)
    assert all(P.is_inside(points))
    P = Polygon2D([[0, 0], [0.3, 0], [0.3, 0.9], [0.5, 0.9], [0.5, 0.85], 
                   [1, 0.85], [1, 0.1], [0.4, 0.1], [0.4, 0], [2, 0], 
                   [2, 1], [0, 1]])
    points = P.sample_inside(50)
    assert np.shape(points) == (50, 2)
    assert all(P.is_inside(points))


def test_lhs_sampling_inside_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    points = P.sample_inside(50, type='lhs')
    assert np.shape(points) == (50, 2)
    assert all(P.is_inside(points))


def test_add_additional_points_if_some_missing_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 0], [10, 10]])
    T = triangulate(P.polygon)[0]
    points =np.ones((4, 2))
    n = 4
    points = P._check_enough_points_sampled(n, points, T)
    assert np.shape(points) == (4, 2)
    n = 8
    points = P._check_enough_points_sampled(n, points, T)
    assert np.shape(points) == (8, 2)
    assert np.all(P.is_inside(points))


def test_bounds_for_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    bounds = P._compute_bounds()
    assert bounds == [0, 10, 0, 10]
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8], [5, 20]])
    bounds = P._compute_bounds()
    assert bounds == [0, 10, 0, 20]


def test_grid_sampling_inside_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = P.sample_inside(250, type='grid')
    assert np.shape(points) == (250, 2)
    assert all(P.is_inside(points))


def test_random_sampling_inside_concav_poly2D():
    np.random.seed(0)
    P = Polygon2D([[0, 0], [0, -5], [-10, -5], [-10, -10], [10, -10],
                   [10, 10], [-10, 10], [-10, 0]])
    points = P.sample_inside(263)
    assert np.shape(points) == (263, 2)
    assert all(P.is_inside(points))


def test_boundary_normal_for_concav_poly2D():
    P = Polygon2D([[0, 0], [0, -10], [10, -10], [10, 10], [-10, 10], [-10, 0]])
    points = [[0, -5], [5, -10], [5, 10], [-10, 7], [-4, 0], [10, 0]]
    normals = P.boundary_normal(points)
    assert np.allclose(normals[0], [-1, 0])
    assert np.allclose(normals[1], [0, -1])
    assert np.allclose(normals[2], [0, 1])
    assert np.allclose(normals[3], [-1, 0])
    assert np.allclose(normals[4], [0, -1])
    assert np.allclose(normals[5], [1, 0])


def test_boundary_normal_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    points = [[0, 5], [10, 5], [1, 2.0/10], [9, 8+2.0/10]]
    normals = P.boundary_normal(points)
    assert np.allclose(normals[0], [-1, 0])
    assert np.allclose(normals[1], [1, 0])
    norm = np.sqrt(2**2+10**2)
    assert np.allclose(normals[3], [2/norm, 10/norm])
    assert np.allclose(normals[2], [2/norm, -10/norm])


def test_boundary_normal_poly2D_with_hole():
    h = s_geo.Polygon(shell=[[0.15, 0.15], [0.25, 0.15], [0.15, 0.25]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    points = [[0.5, 0], [0, 0.5], [0.2, 0.15]]
    normals = P.boundary_normal(points)
    assert np.shape(normals) == (3, 2)
    assert np.allclose(normals[0], [0, -1])
    assert np.allclose(normals[1], [-1, 0])
    assert np.allclose(normals[2], [0, 1])


def test_output_type_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    i_rand = P.sample_inside(1)
    b_rand = P.sample_boundary(1)
    i_grid = P.sample_inside(1, type='grid')
    b_grid = P.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)


def test_outline_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]])
    outline = P.outline()
    assert isinstance(outline, list)
    assert np.allclose(outline[0], [[0, 10], [0, 0], [10, 2], [10, 8], [0, 10]])


def test_outline_poly2D_with_hole():
    h = s_geo.Polygon(shell=[[0.15, 0.15], [0.25, 0.15], [0.15, 0.25]])
    p = s_geo.Polygon(shell=[[0, 0], [1, 0], [0, 1]], holes=[h.exterior.coords])
    P = Polygon2D(shapely_polygon=p)
    outline = P.outline()
    assert isinstance(outline, list)
    assert np.allclose(outline[0], [[0, 0], [1, 0], [0, 1], [0, 0]])
    assert np.allclose(outline[1], [[0.15, 0.15], [0.15, 0.25],
                                    [0.25, 0.15], [0.15, 0.15]])


def test_serialize_poly2D():
    P = Polygon2D([[0, 10], [0, 0], [10, 2], [10, 8]], tol=0.1)
    dct = P.serialize()
    assert dct['dim'] == 2
    assert dct['tol'] == 0.1
    assert dct['name'] == 'Polygon2D'
    assert np.equal(dct['corner_0'], [0, 10]).all()
    assert np.equal(dct['corner_1'], [0, 0]).all()
    assert np.equal(dct['corner_2'], [10, 2]).all()
    assert np.equal(dct['corner_3'], [10, 8]).all()