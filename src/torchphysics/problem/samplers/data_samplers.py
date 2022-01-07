"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""

from .sampler_base import PointSampler
from ..spaces import Points


class DataSampler(PointSampler):
    """A sampler that processes external created data points.

    Parameters
    ----------
    input_data : torchphysics.spaces.Points or dict
        A points object containing the input data for the model.
    output_data : torchphysics.spaces.Points or dict
        The expected model values at the given input data in the
        correct output space.
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