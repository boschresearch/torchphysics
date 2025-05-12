import torch

from ..domain import Domain
from ...spaces import Points


class HyperCube(Domain):
    """ A n-dimensional cube for sampling parameters.

    Parameters
    ----------
    space : Space
        The space of this object.
    lower_bounds : float or array_like
        The lower bound for each dimension of this hypercube. If a single float 
        is passed in, we assume that all dimensions have this value
        as a lower bound.
    upper_bounds : float or array_like
        The upper bound for each dimension of this hypercube. If a single float 
        is passed in, we assume that all dimensions have this value
        as an upper bound.
    
    Note
    ----
    This class is only meant for randomly sampling parameters in high dimensional
    spaces. Currently many functionalites of other domains are not supported in this
    class. This includes, calling `.boundary`, checking if points
    are contained in this HyperCube and also domain operations.
    Additionally, HyperCubes can not dependt on other parameters!
    """
    def __init__(self, space, lower_bounds, upper_bounds):
        super().__init__(space=space, dim=space.dim)
        self.lower_bounds = self._check_bounds(lower_bounds)
        self.upper_bounds = self._check_bounds(upper_bounds)
        self.shifts = self.upper_bounds - self.lower_bounds

        self.volume_tensor = torch.prod(self.shifts, dim=0, keepdim=True).unsqueeze(-1)
        assert self.volume_tensor.item() > 0, "Volume of HyperCube is negative!"

    def _check_bounds(self, bound):
        if isinstance(bound, (float, int)):
            return torch.tensor(float(bound)).repeat(self.dim)
        elif isinstance(bound, (list, tuple)):
            assert len(bound) == self.dim, "Length of bounds does not fit dimension"
            return torch.tensor(bound)
        elif torch.is_tensor(bound):
            assert len(bound.flatten()) == self.dim, "Length of bounds does not fit dimension"
            return bound.flatten()
        else:
            raise ValueError("Type for bounds is not supported!")
        
    def _get_volume(self, params=Points.empty(), device="cpu"):
        return self.volume_tensor.to(device)
    

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), device="cpu"):
        if d: n = self.compute_n_from_density(d, params)        
        num_of_params = self.len_of_params(params)

        points = torch.rand((num_of_params, n, self.dim), device=device)
        points *= self.shifts[None, None, :]
        points += self.lower_bounds[None, None, :]
        return Points(points.reshape(-1, self.space.dim), self.space)
    
    def sample_grid(self, n=None, d=None, params=Points.empty(), device="cpu"):
        if d: n = self.compute_n_from_density(d, params)

        grid_list = []
        scale_n = torch.pow(n / self.volume_tensor, 1.0/self.dim)
        for i in range(self.dim):
            n_i = int(self.shifts[i] * scale_n) + 1
            grid_list.append(torch.linspace(self.lower_bounds[i],
                                            self.upper_bounds[i], 
                                            n_i, device=device)) 
        
        grid = torch.stack(torch.meshgrid(*grid_list))
        permute_idx = torch.flip(torch.arange(self.dim+1), dims=(0,))
        grid = torch.permute(grid, tuple(permute_idx))

        return Points(grid.reshape(-1, self.dim)[:n], self.space)