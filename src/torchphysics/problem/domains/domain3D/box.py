import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points


class Box(Domain):
    """Class for three dimensional boxes.

    Parameters
    ----------
    space : Space
        The space of this object.
    origin : array_like or callable
        The origin of this box (one corner).
    width, height, depth : float or callable
        The size of the box in the three space dimensions.
    """
    def __init__(self, space, origin, width, height, depth):
        assert space.dim == 3
        origin, width, height, depth = self.transform_to_user_functions(
            origin, width, height, depth
        )
        self.origin = origin
        self.width = width
        self.height = height
        self.depth = depth
        super().__init__(space=space, dim=3)
        self.set_necessary_variables(self.origin, self.width, self.height, self.depth)

    def __call__(self, **data):
        new_origin = self.origin.partially_evaluate(**data)
        new_origin = self._check_shape_of_evaluated_user_function(new_origin)
        width = self.width.partially_evaluate(**data)
        height = self.height.partially_evaluate(**data)
        depth = self.depth.partially_evaluate(**data)
        return Box(self.space, new_origin, width, height, depth)

    def _check_shape_of_evaluated_user_function(self, domain_param):
        if isinstance(domain_param, torch.Tensor):
            if len(domain_param.shape) > 1:
                return domain_param[0, :]
        return domain_param
    
    def _eval_box_params(self, params, device):
        eval_origin = self.origin(params, device).reshape(-1, 3)
        eval_width = self.width(params, device).reshape(-1, 1)
        eval_height = self.height(params, device).reshape(-1, 1)
        eval_depth = self.depth(params, device).reshape(-1, 1)     
        return eval_origin, eval_width, eval_height, eval_depth  

    def _get_volume(self, params=Points.empty(), device="cpu"):
        _, eval_width, eval_height, eval_depth = self._eval_box_params(params, device)
        return eval_width * eval_height * eval_depth
    
    def bounding_box(self, params=Points.empty(), device="cpu"):
        eval_origin, eval_width, eval_height, eval_depth = \
            self._eval_box_params(params, device)

        min_values = torch.min(eval_origin, dim=0)[0]
        bounds = []
        scale_list = [eval_width, eval_height, eval_depth]
        for i in range(self.space.dim):
            bounds.append(min_values[i])
            bounds.append(torch.max(eval_origin[:, i:i+1] + scale_list[i]).item())

        return torch.tensor(bounds, device=device)
    
    def _contains(self, points, params=Points.empty()):
        eval_origin, eval_width, eval_height, eval_depth = \
            self._eval_box_params(params, points.device)
        
        points = points[:, list(self.space.keys())].as_tensor
        points -= eval_origin
        inside = torch.ones((len(points), 1), dtype=torch.bool)
        scale_list = [eval_width, eval_height, eval_depth]

        for i in range(self.space.dim):
            in_current = torch.logical_and(0 <= points[:, i:i+1], 
                                           points[:, i:i+1] <= scale_list[i])
            inside = torch.logical_and(in_current, inside)

        return inside.reshape(-1, 1)
    
    def sample_random_uniform(self, n=None, d=None, 
                              params=Points.empty(), device="cpu"):
        if d: n = self.compute_n_from_density(d, params)
        eval_origin, eval_width, eval_height, eval_depth = \
            self._eval_box_params(params, device)
        
        num_of_params = self.len_of_params(params)
        points = torch.rand((num_of_params, n, 3), device=device)
        points[..., 0:1] *= eval_width
        points[..., 1:2] *= eval_height
        points[..., 2:3] *= eval_depth
        points += eval_origin[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)
    
    def sample_grid(self, n=None, d=None, params=Points.empty(), device="cpu"):
        if d:
            n = self.compute_n_from_density(d, params)
        eval_origin, eval_width, eval_height, eval_depth = \
            self._eval_box_params(params, device)

        grid = self._compute_grid(n, eval_width, eval_height, eval_depth, device)

        if not d:
            # if the number of points is specified we have to be sure to sample
            # the right amount
            grid = self._grid_enough_points(n, grid, device)

        scale_list = [eval_width, eval_height, eval_depth]
        for i in range(self.space.dim):
            grid[:, i:i+1] *= scale_list[i]

        grid += eval_origin

        return Points(grid, self.space)

    def _compute_grid(self, n, w, h, d, device):
        volume = w*h*d
        # scale the number of point w.r.t. the shape of the box
        n_scale = torch.pow(n / volume, 1.0/3.0)
        n_z = int(d * n_scale)
        n_x = int(w * n_scale)
        n_y = int(h * n_scale)
        x = torch.linspace(0, 1, n_x + 2, device=device)[1:-1] 
        y = torch.linspace(0, 1, n_y + 2, device=device)[1:-1]
        z = torch.linspace(0, 1, n_z + 2, device=device)[1:-1]
        grid = torch.permute(torch.stack(torch.meshgrid((x, y, z))), (3, 2, 1, 0))
        return grid.reshape(-1, 3)

    def _grid_enough_points(self, n, grid, device):
        if len(grid) < n:
            random_points = torch.rand((n - len(grid), 3), device=device)
            grid = torch.cat((grid, random_points), dim=0)
        return grid
    
    @property
    def boundary(self):
        return BoxBoundary(self)


class BoxBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, Box)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        eval_origin, eval_width, eval_height, eval_depth = \
            self.domain._eval_box_params(params, points.device)
        
        points = points[:, list(self.space.keys())].as_tensor
        points -= eval_origin

        on_boundary = torch.zeros((len(points), 1), dtype=torch.bool)
        scale_list = [eval_width, eval_height, eval_depth]

        for i in range(self.space.dim):      
            # check current dir. on boundary
            close_0 = torch.isclose(points[:, i:i+1], 
                                    torch.tensor(0.0, device=points.device))
            close_size = torch.isclose(points[:, i:i+1], scale_list[i])
            # combine
            on_boundary |= (close_0 | close_size)

        # also need to check if points are in the box (one dimension is on the 
        # boundary so the other dimension need to lay inside)
        in_current = \
            torch.logical_and(0 <= points[:, 0:1], points[:, 0:1] <= scale_list[0]) &\
            torch.logical_and(0 <= points[:, 1:2], points[:, 1:2] <= scale_list[1]) &\
            torch.logical_and(0 <= points[:, 2:3], points[:, 2:3] <= scale_list[2])

        on_boundary &= in_current

        return on_boundary.reshape(-1, 1)
    
    def _get_volume(self, params=Points.empty(), device="cpu"):
        _, eval_width, eval_height, eval_depth = \
            self.domain._eval_box_params(params, device=device)
        
        area_xy = eval_width * eval_height
        area_xz = eval_width * eval_depth
        area_yz = eval_depth * eval_height
        return 2 * (area_xy + area_xz + area_yz).reshape(-1, 1)
    
    def sample_random_uniform(self, n=None, d=None, 
                              params=Points.empty(), device="cpu"):
        if d:
            n = self.compute_n_from_density(d, params)
        eval_origin, eval_width, eval_height, eval_depth = \
            self.domain._eval_box_params(params, device=device)
        
        # Amount of points need to scale with surface area
        area_xy = eval_width * eval_height
        area_xz = eval_width * eval_depth
        area_yz = eval_depth * eval_height
        total_area = (area_xy + area_xz + area_yz).reshape(-1, 1)
        
        area_list = [area_yz, area_xz, area_xy]
        scale_list = [eval_width, eval_height, eval_depth]
        
        num_of_params = self.len_of_params(params)
        points = torch.zeros((num_of_params, n, 3), device=device)
        indices = torch.arange(n).view(1, n)
        old_n_scale = torch.zeros_like(area_xy)
        
        # We iterate over each direction and sample on each side seperately.
        for i in range(self.space.dim):
            if i < self.space.dim - 1:
                n_scale = int(n * area_list[i] / total_area)
            else: # last dimension gets all remaining points
                n_scale = n - old_n_scale

            # Note: we sample total amount of n points, since the domain
            # could be dependent on other parameters which leads to different
            # scaling of the surface area and therefore to different amount 
            # of points depending on the parameters.
            # To handle all cases we sample more points and then override
            # only the needed points.    
            current_points = torch.rand((num_of_params, n, 3), device=device) 
            current_shift  = torch.randint(0, 2, (num_of_params, n, 1), device=device) 
            
            current_points[:, :, i:i+1] = current_shift * scale_list[i]
            i_mod_1 = (i+1)%3
            i_mod_2 = (i+2)%3
            current_points[:, :, i_mod_1:i_mod_1+1] *= scale_list[i_mod_1]
            current_points[:, :, i_mod_2:i_mod_2+1] *= scale_list[i_mod_2]

            mask = torch.logical_or(indices >= n_scale+old_n_scale, indices < old_n_scale)
            # set not needed values to 0
            current_points[mask.unsqueeze(-1).repeat((1, 1, 3))] = 0.0
            
            old_n_scale += n_scale
            points += current_points

        points += eval_origin[:, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)
    
    def sample_grid(self, n=None, d=None, params=Points.empty(), device="cpu"):
        if d:
            n = self.compute_n_from_density(d, params)
        eval_origin, eval_width, eval_height, eval_depth = \
            self.domain._eval_box_params(params, device=device)
        
        # Amount of points need to scale with surface area
        area_xy = eval_width * eval_height
        area_xz = eval_width * eval_depth
        area_yz = eval_depth * eval_height
        total_area = (area_xy + area_xz + area_yz).reshape(-1, 1)
        
        area_list = [area_yz, area_xz, area_xy]
        scale_list = [eval_width, eval_height, eval_depth]
        permute_list = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        difference_list = [[eval_height, eval_depth], 
                           [eval_depth, eval_width], 
                           [eval_width, eval_height]]
        points = torch.zeros((n, 3), device=device)
        current_n = 0
        
        # We iterate over each direction and sample on each side seperately.
        for i in range(self.space.dim):
            if i < self.space.dim - 1:
                n_scale = int(n * area_list[i] / total_area)
            else: # last dimension gets all remaining points
                n_scale = n - current_n
            
            n_1 = int(torch.sqrt(n_scale/2.0 * difference_list[i][0]/difference_list[i][1]))
            n_2 = int(torch.sqrt(n_scale/2.0 * difference_list[i][1]/difference_list[i][0]))
            
            grid_1 = torch.linspace(0, 1, n_1+1, device=device)
            grid_2 = torch.linspace(0, 1, n_2+1, device=device)
            # Bottom and top side, always create a grid that contains two edges.
            # When we stick all sides together we obtain therefore all edges
            for k in range(2):
                n_prod = n_1 * n_2
                if k == 0:
                    grid = torch.permute(
                        torch.stack(torch.meshgrid((grid_1[:-1], grid_2[:-1]))), (2, 1, 0)
                        ).reshape(-1, 2)
                    grid = torch.column_stack(
                        (torch.zeros((len(grid), 1), device=device), grid)
                        )
                else:
                    grid = torch.permute(
                        torch.stack(torch.meshgrid((grid_1[1:], grid_2[1:]))), (2, 1, 0)
                        ).reshape(-1, 2)
                    # add some random points if no perfect grid was created:
                    n_difference = n_scale - (n_prod + len(grid))
                    if n_difference > 0:
                        random_points = torch.rand((n_difference, 2), device=device)
                        grid = torch.cat((grid, random_points), dim=0)
                        n_prod = len(grid)

                    grid = torch.column_stack(
                        (scale_list[i] * torch.ones((len(grid), 1), device=device), grid)
                        )
                
                i_mod_1 = (i+1)%3
                i_mod_2 = (i+2)%3
                grid[:, 1:2] *= scale_list[i_mod_1]
                grid[:, 2:3] *= scale_list[i_mod_2]

                grid = grid[:, permute_list[i]]
                points[current_n:current_n+n_prod] = grid

                current_n += n_prod

        points += eval_origin
        return Points(points.reshape(-1, self.space.dim), self.space)

    def normal(self, points, params=Points.empty(), device="cpu"):
        points, params, device = self._transform_input_for_normals(
            points, params, device
        )
        eval_origin, eval_width, eval_height, eval_depth = \
            self.domain._eval_box_params(params, device=device)
        
        points = points[:, list(self.space.keys())].as_tensor
        points -= eval_origin

        normals = torch.zeros_like(points, device=device)
        scale_list = [eval_width, eval_height, eval_depth]

        for i in range(self.space.dim):      
            # check current dir. on boundary
            close_0 = torch.isclose(points[:, i:i+1], 
                                    torch.tensor(0.0, device=points.device)).squeeze(-1)
            close_size = torch.isclose(points[:, i:i+1], scale_list[i]).squeeze(-1)
            normals[close_0, i] = -1.0
            normals[close_size, i] = 1.0

        # scale normal vectors if they are in a corner:
        return torch.divide(normals, torch.linalg.norm(normals, dim=1).reshape(-1, 1))