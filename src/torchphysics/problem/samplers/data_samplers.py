"""File with samplers that handle external created data.
E.g. measurements or validation data computed with other methods.
"""
import torch

from .sampler_base import PointSampler
from ..spaces import Points

import time

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
        n = len(self.points.as_tensor)
        super().__init__(n_points=n)

    def sample_points(self, params=Points.empty(), device="cpu"):
        self.points = self.points.to(device)
        
        # If sampler not coupled to other samplers or parameters
        # we can return:
        if params.isempty:
            return self.points

        # Maybe given data has more dimensions than batch and space
        # (For example evaluation on quadrature points)
        # TODO: Make more general. What happends when parameters have higher dimension?
        # What when multiple dimension in both that do not fit?
        start_time = time.time()
        if len(self.points.as_tensor.shape) > 2:
            repeated_tensor = params.as_tensor
            for i in range(1, len(self.points.as_tensor.shape)-1):
                repeated_tensor = torch.repeat_interleave(repeated_tensor.unsqueeze(-1),
                                                        self.points.as_tensor.shape[i],
                                                        dim=i)
            
            repeated_params = Points(repeated_tensor, params.space)
        print("Dimension thing took", time.time() - start_time)

        # else we have to repeat data (meshgrid of both) and join the tensors together:
        start_time = time.time()
        repeated_params = self._repeat_params(repeated_params, len(self))
        print("Repeating params took", time.time() - start_time)
        start_time = time.time()
        repeated_points = self.points.repeat(len(params))
        print("Repeating points took", time.time() - start_time)

        return repeated_points.join(repeated_params)