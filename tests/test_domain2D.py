import pytest
import numpy as np
from neural_diff_eq.problem.domain.domain2D import Rectangle, Circle

# Tests for rectangle

def test_create_rectangle():
    R = Rectangle([0,0],[1,0],[0,1])

def test_throw_error_if_not_rectangle():
    with pytest.raises(ValueError):
        R = Rectangle([0,0],[2,0],[2,0])

def test_check_dim_rect():
    R = Rectangle([0,0],[1,0],[0,1])
    assert R.dim == 2

def test_check_corners_rect():
    R = Rectangle([0,0],[1,0],[0,1])
    assert all((R.corner_dl == [0,0]))
    assert all((R.corner_dr == [1,0]))
    assert all((R.corner_tl == [0,1]))

def test_check_side_length_rect():
    R = Rectangle([0,0],[1,0],[0,2])
    assert R.length_lr == 1
    assert R.length_td == 2

def test_check_inverse_matrix_rect():
    R = Rectangle([0,0],[1,0],[0,2])
    assert R.inverse_matrix[0][0] == 1
    assert R.inverse_matrix[0][1] == 0
    assert R.inverse_matrix[1][0] == 0
    assert R.inverse_matrix[1][1] == 1/2

def test_output_type_rect():
    R = Rectangle([0,0],[1,0],[0,1])
    i_rand = R.sample_inside(1)
    b_rand = R.sample_boundary(1)
    i_grid = R.sample_inside(1, type='grid')
    b_grid = R.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)

def test_check_points_inside_and_outside_rect():
    R = Rectangle([1,0],[2,1],[2,-1])
    points = [[2,0],[0,0]]
    inside = R.is_inside(points)
    assert inside[0]
    assert not inside[1]

def test_check_points_boundary_rect():
    R = Rectangle([0,0],[1,0],[0,3])
    points = [[0,0.2],[1,0],[0.5,0.5],[-2,10],[0,-0.1]]
    bound = R.is_on_boundary(points)
    assert all(bound[0:1])
    assert not all(bound[2:])

def test_random_sampling_inside_rect():
    np.random.seed(0)
    R = Rectangle([0,0],[1,1],[-2,2])
    points = R.sample_inside(10, type='random')
    compare = [[-1.0346366 , 2.1322637 ], [-0.34260046, 1.7729793 ], [-0.53332573, 1.7388525 ],
               [-1.30631   , 2.3960764 ], [ 0.28158268, 0.56572694], [ 0.47163552, 0.8201527 ],
               [ 0.39715043, 0.478024  ], [-0.7734667 , 2.5570128 ], [-0.5926508 , 2.5199764 ],
               [-1.3565828 , 2.1234658 ]]
    compare = np.array(compare).astype(np.float32)
    assert len(points) == 10
    assert all(R.is_inside(points))
    assert all((compare == points).all(axis=1))

def test_grid_sampling_inside_rect():
    R = Rectangle([0,0],[2,1],[-1,2])
    points = R.sample_inside(250, type='grid')
    assert len(points) == 250
    assert all(R.is_inside(points))
    assert np.linalg.norm(points[0]-points[1]) == np.linalg.norm(points[2]-points[1])

def test_grid_sampling_inside_append_center_rect():
    R = Rectangle([1,0],[2,-1],[2,1])
    points = R.sample_inside(27, type='grid')
    assert len(points) == 27
    assert points[-1][0] == 2
    assert points[-1][1] == 0

def test_random_sampling_boundary_rect():
    np.random.seed(0)
    R = Rectangle([0,0],[2,1],[-1,2]) 
    points = R.sample_boundary(10, type='random')
    compare = [[ 0.09762701, 2.5488136 ], [ 1.4303787 , 0.71518934], [ 1.2055267 , 0.60276335],
               [ 0.08976637, 2.5448833 ], [ 0.8473096 , 0.4236548 ], [ 1.943287  , 1.113426  ],
               [ 1.7273437 , 1.5453126 ], [ 1.5223349 , 1.9553303 ], [ 1.1878313 , 2.6243374 ],
               [-0.47997716, 0.9599543 ]]
    compare = np.array(compare).astype(np.float32)
    assert len(points) == 10
    assert all(R.is_on_boundary(points))
    assert all((compare == points).all(axis=1))

def test_grid_sampling_boundary_rect():
   R = Rectangle([0,0],[2,1],[-1,2]) 
   points = R.sample_boundary(150, type='grid')
   assert len(points) == 150
   assert all(R.is_on_boundary(points))
   assert np.linalg.norm(points[0]-points[1]) == np.linalg.norm(points[2]-points[1])

def test_boundary_normal_rect():
    R = Rectangle([0,0],[2,0],[0,2])
    point = [[0,0.5],[0.7,2]]
    normal = R.boundary_normal(point)
    assert normal[0][0] == -1
    assert normal[0][1] == 0
    assert normal[1][0] == 0
    assert normal[1][1] == 1

def test_normal_vector_at_corner_():
    R = Rectangle([0,0],[2,0],[0,2])
    point = [[0,0],[2,0],[0,2],[2,2]]
    normal = R.boundary_normal(point)
    for i in range(4):
        assert np.isclose(np.linalg.norm(normal[i]),1)
        
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
    R = Rectangle([1,0],[2,-1],[2,1])
    point = [[1.5,-0.5], [1.5,0.5],[1.5,1.5]]
    normal = R.boundary_normal(point)
    assert np.isclose(np.dot([1,-1], normal[0]), 0)
    assert np.isclose(np.dot([1,1], normal[1]), 0)
    assert np.isclose(np.dot([1,-1], normal[2]), 0)

def test_console_output_when_not_on_bound_rect(capfd):
    R = Rectangle([0,0],[9,0],[0,5])
    point = [[3,3]]
    R.boundary_normal(point)
    out, err = capfd.readouterr() 
    assert out == 'Warninig: some points are not at the boundary!\n'

# Test for circle
def test_create_circle():
    C = Circle([0,0],2)

def test_check_dim_circle():
    C = Circle([0,0],2)
    assert C.dim == 2

def test_check_center_and_radius():
    C = Circle([0.5,0.6], 2)
    assert C.center[0] == 0.5
    assert C.center[1] == 0.6
    assert C.radius == 2

def test_output_type_circle():
    C = Circle([1,0],3)
    i_rand = C.sample_inside(1)
    b_rand = C.sample_boundary(1)
    i_grid = C.sample_inside(1, type='grid')
    b_grid = C.sample_boundary(1, type='grid')
    assert isinstance(i_rand[0][0], np.float32)
    assert isinstance(b_rand[0][0], np.float32)
    assert isinstance(i_grid[0][0], np.float32)
    assert isinstance(b_grid[0][0], np.float32)

def test_check_points_inside_and_outside_circle():
    C = Circle([1,0],2)
    points = [[2,0],[1,1],[10,0]]
    inside = C.is_inside(points)
    assert all(inside[0:2])
    assert not inside[2]

def test_check_points_boundary_circle():
    C = Circle([1,0],2)
    points = [[2,0],[1,1],[3,0],[1,2]]
    inside = C.is_on_boundary(points)
    assert not all(inside[0:2])
    assert all(inside[2:])

def test_random_sampling_inside_circle():
    np.random.seed(0)
    C = Circle([1,1],3)
    points = C.sample_inside(5, type='random')
    compare = [[-0.35227352, -0.7637114 ], [-1.344475  ,  1.9696087 ], [ 2.8110569 , -0.46456257],
               [ 3.157019  ,  0.4987838 ], [-0.4519993 ,  2.3056    ]]
    compare = np.array(compare).astype(np.float32)
    assert len(points) == 5
    assert all(C.is_inside(points))
    assert all((compare == points).all(axis=1))

def test_grid_sampling_inside_circle():
    C = Circle([0,0], 4)
    points = C.sample_inside(250, type='grid')
    assert len(points) == 250
    assert all(C.is_inside(points))

def test_grid_sampling_inside_append_center_circle():
    C = Circle([0,1], 4)
    points = C.sample_inside(27, type='grid')
    assert len(points) == 27
    assert C.center[0] == points[-1][0]
    assert C.center[1] == points[-1][1]

def test_random_sampling_boundary_circle():
    np.random.seed(0)
    C = Circle([0,0],4.5) 
    points = C.sample_boundary(10, type='random')
    compare = [[-4.290002 , -1.3586327], [-0.9764186, -4.3927903], [-3.5941048, -2.7078424], 
               [-4.322242 , -1.2522879], [-3.992119 ,  2.076773 ], [-2.7380629, -3.571136 ],
               [-4.158401 ,  1.719797 ], [ 3.4990482, -2.8296046], [ 4.3832226, -1.0185084],
               [-3.3461978,  3.0088139]]
    compare = np.array(compare).astype(np.float32)
    assert len(points) == 10
    assert all(C.is_on_boundary(points))
    assert all((compare == points).all(axis=1))

def test_grid_sampling_boundary_circle():
    C = Circle([0,0],2) 
    points = C.sample_boundary(150, type='grid')
    assert len(points) == 150
    assert all(C.is_on_boundary(points))
    for i in range(len(points)-2):
        assert np.isclose(np.linalg.norm(points[i]-points[i+1]),
                          np.linalg.norm(points[i+2]-points[i+1]))

def test_boundary_normal_circle():
    C = Circle([1,0],2)
    point = [[3,0],[1,2]]
    normal = C.boundary_normal(point)
    assert normal[0][0] == 1
    assert normal[0][1] == 0
    assert normal[1][0] == 0
    assert normal[1][1] == 1

def test_normal_vector_length_circle():
    C = Circle([0,0],5)
    points = C.sample_boundary(50)
    normal = C.boundary_normal(points)
    for i in range(len(points)):
        assert np.isclose(np.linalg.norm(normal[i]),1)

def test_console_output_when_not_on_bound_circle(capfd):
    C = Circle([0,0],5)
    point = [[0,0]]
    C.boundary_normal(point)
    out, err = capfd.readouterr() 
    assert out == 'Warninig: some points are not at the boundary!\n'