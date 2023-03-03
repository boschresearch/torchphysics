import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points

class Triangle(Domain):
    '''Class for triangles.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    origin, corner_1, corner_2 : array_like or callable
        The three corners of the triangle. The corners have to be ordered counter
        clockwise, to assure that the normal vectors will point outwards.
    '''
    def __init__(self, space, origin, corner_1, corner_2):
        assert space.dim == 2
        origin, corner_1, corner_2 = \
            self.transform_to_user_functions(origin, corner_1, corner_2)
        self.origin = origin
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        super().__init__(space=space, dim=2)
        self.set_necessary_variables(self.origin, self.corner_1, self.corner_2)

    def __call__(self, **data):
        new_origin = self.origin.partially_evaluate(**data)
        new_origin = self._check_shape_of_evaluated_user_function(new_origin)
        new_vec_1 = self.corner_1.partially_evaluate(**data)
        new_vec_1 = self._check_shape_of_evaluated_user_function(new_vec_1)
        new_vec_2 = self.corner_2.partially_evaluate(**data)
        new_vec_2 = self._check_shape_of_evaluated_user_function(new_vec_2)
        return Triangle(space=self.space, origin=new_origin, 
                        corner_1=new_vec_1, corner_2=new_vec_2)

    def _check_shape_of_evaluated_user_function(self, domain_param):
        if isinstance(domain_param, torch.Tensor):
            if len(domain_param.shape) > 1:
                return domain_param[0, :]
        return domain_param

    def _get_volume(self, params=Points.empty(), device='cpu'):
        _, _, _, dir_1, _, dir_3 = self._construct_triangle(params, device=device)
        # volume equals the determinate of the matrix [dir_1, dir_2] / 2
        volume = -dir_1[:, :1] * dir_3[:, 1:] + dir_1[:, 1:] * dir_3[:, :1]
        return volume / 2.0

    def _construct_triangle(self, params=Points.empty(), device='cpu'):
        origin = self.origin(params, device).reshape(-1, 2)
        corner_1 = self.corner_1(params, device).reshape(-1, 2)
        corner_2 = self.corner_2(params, device).reshape(-1, 2)
        dir_1 = corner_1 - origin
        dir_2 = corner_2 - corner_1
        dir_3 = origin - corner_2
        return origin, corner_1, corner_2, dir_1, dir_2, dir_3

    def bounding_box(self, params=Points.empty(), device='cpu'):
        origin, corner_1, corner_2, _, _, _ = self._construct_triangle(params, device=device)
        bounds = []
        for i in range(self.dim):
            dim_i_max, dim_i_min = [], []
            for corner in [origin, corner_1, corner_2]:
                dim_i_max.append(torch.max(corner[:, i]).item())
                dim_i_min.append(torch.min(corner[:, i]).item())
            bounds.append(min(dim_i_min))
            bounds.append(max(dim_i_max))
        return torch.tensor(bounds, device=device)

    def _contains(self, points, params=Points.empty()):
        origin, _, _, dir_1, _, dir_3 = \
            self._construct_triangle(points.join(params), device=points.device)
        points = points[:, list(self.space.keys())].as_tensor
        points -= origin
        bary_x, bary_y = self._solve_lgs(points, dir_1, -dir_3)
        greater_0 = torch.logical_and(0 <= bary_x, 0 <= bary_y)
        sum_smaller_1 = (bary_y + bary_x) <= 1
        return torch.logical_and(greater_0, sum_smaller_1).reshape(-1, 1)

    def _solve_lgs(self, points, dir_1, dir_2):
        # To check if points are inside the triangle we solve the
        # linear system [dir_1, dir_2] * [x, y] = point, for each point.
        # We solve this with the inverse and all points at the same time:
        det = dir_1[:, :1] * dir_2[:, 1:] - dir_1[:, 1:] * dir_2[:, :1]
        x_dir = dir_2[:, 1:] * points[:, :1] - dir_2[:, :1] * points[:, 1:]
        y_dir = dir_1[:, :1] * points[:, 1:] - dir_1[:, 1:] * points[:, :1]
        bary_x = torch.divide(x_dir, det)
        bary_y = torch.divide(y_dir, det)
        return bary_x, bary_y

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if d:
            n = 2*self.compute_n_from_density(d, params)
        origin, _, _, dir_1, _, dir_3 = self._construct_triangle(params, device)
        num_of_params = self.len_of_params(params)
        bary_coords = torch.rand((num_of_params, n, 2), device=device)
        bary_coords = self._handle_sum_greater_1(d, bary_coords)
        points_in_dir_1 = bary_coords[:, :, :1] * dir_1[:, None]
        points_in_dir_2 = -bary_coords[:, :, 1:] * dir_3[:, None]
        points = points_in_dir_1 + points_in_dir_2
        points += origin[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def _handle_sum_greater_1(self, d, bary_coords):
        sum_bigger_one = (bary_coords.sum(axis=2) >= 1)
        if d: # for a given density just remove the points 
            index = torch.where(torch.logical_not(sum_bigger_one))
            return bary_coords[index][None, :]
        # for a given number of points, we want the correct number. 
        # So mirror the points, with sum greater 1, around the point [0.5, 0.5].
        # This stays uniform.
        index = torch.where(sum_bigger_one)
        bary_coords[index] = torch.subtract(torch.tensor([[1.0, 1.0]]), 
                                            bary_coords[index])
        return bary_coords

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        origin, _, _, dir_1, _, dir_3 = self._construct_triangle(params, device)
        bary_coords = self._compute_barycentric_grid(n, dir_1, dir_3, device)
        if not d: 
            # if the number of points is specified we have to be sure to sample
            # the right amount 
            bary_coords = self._grid_has_n_points(n, bary_coords, device)
        points_in_dir_1 = bary_coords[:, :1] * dir_1
        points_in_dir_2 = -bary_coords[:, 1:] * dir_3
        points = points_in_dir_1 + points_in_dir_2
        points += origin
        return Points(points, self.space)

    def _compute_barycentric_grid(self, n, dir_1, dir_2, device):
        # since we have a triangle, we rmove later the points with bary.-
        # coordinates bigger one. Therefore double the number of points:
        scaled_n = 2 * n
        side_length_1 = torch.linalg.norm(dir_1, dim=1)
        side_length_2 = torch.linalg.norm(dir_2, dim=1)
        # scale the number of point w.r.t. the 'form' of the parallelogram
        n_1 = int(torch.sqrt(scaled_n*side_length_1/side_length_2))
        n_2 = int(torch.sqrt(scaled_n*side_length_2/side_length_1))
        x = torch.linspace(0, 1, n_1+2, device=device)[1:-1] # create inner grid, so remove
        y = torch.linspace(0, 1, n_2+2, device=device)[1:-1] # first and last value
        bary_coords = torch.stack(torch.meshgrid((x, y))).T.reshape(-1, 2)
        index = torch.where(bary_coords.sum(axis=1) <= 1)
        return bary_coords[index]

    def _grid_has_n_points(self, n, bary_coords, device): 
        # if not enough points, add some random ones.
        if len(bary_coords) < n:
            random_points = torch.rand((n - len(bary_coords), 2), device=device)
            index = torch.where(random_points.sum(axis=1) >= 1)
            random_points[index] = torch.subtract(torch.tensor([[1.0, 1.0]]), 
                                                  random_points[index])
            bary_coords = torch.cat((bary_coords, random_points), dim=0)
        elif len(bary_coords) > n:
            # just take the first n elements
            bary_coords = bary_coords[:n] 
        return bary_coords

    @property
    def boundary(self):
        return TriangleBoundary(self)


class TriangleBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Triangle)
        super().__init__(domain)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        _, _, _, dir_1, dir_2, dir_3 = self.domain._construct_triangle(params, device=device)
        side_1, side_2, side_3 = self._compute_side_length(dir_1, dir_2, dir_3)
        side_length = side_1 + side_2 + side_3
        return side_length.reshape(-1, 1)

    def _contains(self, points, params=Points.empty()):
        origin, _, _, dir_1, _, dir_3 = \
            self.domain._construct_triangle(points.join(params), device=points.device)
        points = points[:, list(self.space.keys())].as_tensor
        points -= origin
        bary_x, bary_y = self.domain._solve_lgs(points, dir_1, -dir_3)
        x_close_to_0 = self._bary_coords_close_to_0_or_1(bary_x, bary_y)
        y_close_to_0 = self._bary_coords_close_to_0_or_1(bary_y, bary_x)
        sum_close_to_1 = torch.isclose(bary_x + bary_y, torch.tensor(1.0))
        close_to_0 = torch.logical_or(x_close_to_0, y_close_to_0)
        return torch.logical_or(close_to_0, sum_close_to_1).reshape(-1, 1)

    def _bary_coords_close_to_0_or_1(self, bary_coord1, bary_coord2):
        between_0_1 = torch.logical_and(0 <= bary_coord2, bary_coord2 <= 1)
        close_to_0 = torch.isclose(bary_coord1, torch.tensor(0.0))
        return torch.logical_and(close_to_0, between_0_1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        # general idea is the same as in the parallelogram class. 
        if d:
            n = self.compute_n_from_density(d, params)
        origin, _, _, dir_1, dir_2, dir_3 = self.domain._construct_triangle(params, device)
        side_1, side_2, side_3 = self._compute_side_length(dir_1, dir_2, dir_3)
        total_length = side_1 + side_2 + side_3
        num_of_params = self.len_of_params(params)
        points = torch.zeros((num_of_params, n, 2), device=device)
        bound_location = torch.rand((num_of_params, n, 1), device=device)*total_length
        self._transform_interval_to_boundary(dir_1, dir_2, dir_3, 
                                             side_1, side_2, side_3, 
                                             points, bound_location)
        points += origin[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def _compute_side_length(self, dir_1, dir_2, dir_3):
        side_length_1 = torch.linalg.norm(dir_1, dim=1).view(-1, 1, 1)
        side_length_2 = torch.linalg.norm(dir_2, dim=1).view(-1, 1, 1)
        side_length_3 = torch.linalg.norm(dir_3, dim=1).view(-1, 1, 1)
        return side_length_1, side_length_2, side_length_3

    def _transform_interval_to_boundary(self, dir_1, dir_2, dir_3, 
                                        side_1, side_2, side_3, 
                                        points, bound_location):
        self._scale_points_on_side(dir_1, side_1, points, bound_location)
        self._scale_points_on_side(dir_2, side_2, points, bound_location)
        self._scale_points_on_side(dir_3, side_3, points, bound_location)

    def _scale_points_on_side(self, dir, side_length, points, bound_location):
        scale =  torch.clamp(bound_location/side_length, min=0, max=1)
        points += scale * dir[:, None]
        bound_location -= side_length

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        if d:
            n = self.compute_n_from_density(d, params)
        origin, _, _, dir_1, dir_2, dir_3 = self.domain._construct_triangle(params, device)
        side_1, side_2, side_3 = self._compute_side_length(dir_1, dir_2, dir_3)
        total_length = side_1 + side_2 + side_3
        num_of_params = self.len_of_params(params)
        points = torch.zeros((num_of_params, n, 2), device=device)
        bound_grid = torch.linspace(0, 1, n+1, device=device)[:-1] # last point would be double
        bound_grid = bound_grid.repeat(num_of_params).view(num_of_params, n, 1) 
        bound_location = bound_grid * total_length
        self._transform_interval_to_boundary(dir_1, dir_2, dir_3, 
                                             side_1, side_2, side_3, 
                                             points, bound_location)
        points += origin[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        origin, _, _, dir_1, dir_2, dir_3 = \
            self.domain._construct_triangle(points.join(params), device)
        points = points[:, list(self.space.keys())].as_tensor
        normals = torch.zeros_like(points, device=device)
        bary_x, bary_y = self.domain._solve_lgs(points - origin, dir_1, -dir_3)
        normal_dir_1 = self._get_normal_direction(dir_1, device)
        normal_dir_2 = self._get_normal_direction(dir_2, device)
        normal_dir_3 = self._get_normal_direction(dir_3, device)
        # compute for each point what the normal vector should be, by checking the
        # value of the local barycentric coordinate = 0 or sum = 1
        self._add_local_normal_vector(normals, bary_x, normal_dir_3, 0.0)
        self._add_local_normal_vector(normals, (bary_x + bary_y), normal_dir_2, 1.0)
        self._add_local_normal_vector(normals, bary_y, normal_dir_1, 0.0)
        # scale normal vectors if there where in a corner:
        return torch.divide(normals, torch.linalg.norm(normals, dim=1).reshape(-1, 1))

    def _add_local_normal_vector(self, normals, bary_coord, normal, i):
        close_to_i = torch.where(torch.isclose(bary_coord, torch.tensor(i)), 1.0, 0.0)
        normals += normal * close_to_i

    def _get_normal_direction(self, direction, device):
        # to get normal vector in 2d switch x and y coordinate and multiply 
        # one coordinate with -1
        normal = torch.index_select(direction, 1,
                                    torch.tensor([1, 0], device=device))
        normal[:, 1:] *= -1
        return torch.divide(normal, torch.linalg.norm(normal, dim=1).reshape(-1, 1))