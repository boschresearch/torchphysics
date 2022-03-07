"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""

from .sampler_base import PointSampler
from ..spaces import Points


class DataSampler(PointSampler):
    """A sampler that processes external created data points.

    Parameters
    ----------
    points : torchphysics.spaces.points or dict
        The data points that this data sampler should pass to a condition.
        Either already a torchphysics.spaces.points object or in form of
        dictionary like: {'x': tensor_for_x, 't': tensor_for_t, .....}.
        For the dicitionary all tensor need to have the same batch dimension.
    """

    def __init__(self, points):
        if isinstance(points, Points):
            self.points = points
        elif isinstance(points, dict):
            self.points = Points.from_coordinates(points)
        else:
            raise TypeError("points should be one of Points or dict.")
        n = len(points)
        super().__init__(n_points=n)

    def sample_points(self, params=Points.empty(), device='cpu'):
        self.points = self.points.to(device)
        return self.points