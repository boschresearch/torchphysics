import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Adds positional information to data provided on a uniform grid.
    The input data is expected to have shape (batch, axis_1, ..., axis_n, channels),
    and the output will have shape (batch, axis_1, ..., axis_n, channels + n).

    Parameters
    ----------
    dim : int
        The dimension of the underlying space for which positional
        information should be appended.
    coordinate_boundaries : list, optional
        Boundaries of the underlying grid for each dimension. Expected to be a list
        of lists, where each inner list contains the bounds for a dimension.
        The default is [[0, 1]] * dim.
    
    Notes
    -----
    The positional information is generated on the fly, depending on the input
    shape. This makes the embedding resolution-independent. Since this class is mainly
    used in connection with Fourier Neural Operators (FNOs), a uniform grid is created
    in each direction.
    """

    def __init__(self, dim, coordinate_boundaries=None):
        super().__init__()
        self.dim = dim
        if coordinate_boundaries is not None:
            assert self.dim == len(coordinate_boundaries), \
                f"Dimension is {self.dim} and does not fit provided coordinates of shape {len(coordinate_boundaries)}!"
            self.bounds = coordinate_boundaries
        else:
            self.bounds = [[0, 1]] * self.dim
        self.register_buffer("_positions", torch.empty(0))


    def forward(self, points):
        input_shape = points.as_tensor.shape
        # If we have a new shape of the data, we need to create a new positional embedding
        if not input_shape[1:-1] == self._positions.shape[1:-1]:
            self._build_positional_embedding(input_shape, points.device, points.as_tensor.dtype)
            
        repeated_embedding = self._positions.repeat(input_shape[0], *[1] * (self.dim+1))
        return torch.cat((points.as_tensor, repeated_embedding), dim=-1)
    

    def _build_positional_embedding(self, data_shape, device="cpu", dtype=torch.float32):
        coordinate_grid = []
        for i in range(self.dim):
            coordinate_grid.append(
                torch.linspace(self.bounds[i][0], self.bounds[i][1], 
                               data_shape[i+1], dtype=dtype, device=device)
            )
        coordinate_meshgrid = torch.meshgrid(*coordinate_grid, indexing='ij')
        self._positions = torch.cat([x.unsqueeze(-1) for x in coordinate_meshgrid], dim=-1)
        self._positions = self._positions.unsqueeze(0)