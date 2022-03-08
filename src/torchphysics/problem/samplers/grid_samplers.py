"""File with samplers that create points with some kind of ordered structure.
"""
import torch
import warnings

from .sampler_base import PointSampler
from ..domains.domain1D import Interval
from .random_samplers import RandomUniformSampler
from ..spaces import Points


class GridSampler(PointSampler):
    """Will sample a equidistant point grid in the given domain.

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desiered density of the created points.
    filter_fn : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return false. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """
    def __init__(self, domain, n_points=None, density=None, filter_fn=None):
        super().__init__(n_points=n_points, density=density, filter_fn=filter_fn)
        self.domain = domain

    def _sample_points(self, params=Points.empty(), device='cpu'):
        if any(var in self.domain.necessary_variables for var in params.space.keys()):
            return self._sample_params_dependent(self.domain.sample_grid, params, device)
        return self._sample_params_independent(self.domain.sample_grid, params, device)

    def _sample_points_with_filter(self, params=Points.empty(), device='cpu'):
        if self.n_points:
            sample_points = self._sample_n_points_with_filter(params, device)
        else:
            # for density sampling, just sample normally and afterwards remove all 
            # points that are not allowed
            sample_points = self._sample_points(params, device)
            sample_points = self._apply_filter(sample_points)
        return sample_points

    def _sample_n_points_with_filter(self, params, device):
        # The idea is to first sample normally, then see how many points are valid.
        # Then rescale the number of points to get a better grid and sample again.
        # If still some points are missing add random points.
        sample_function = self.domain.sample_grid
        num_of_params = max(1, len(params))
        sample_points = None
        for i in range(num_of_params):
            ith_params = params[i, ] if len(params) > 0 else Points.empty()
            new_points = self._sample_grid(ith_params, sample_function,
                                           self.n_points, device)
            new_better_points = self._resample_grid(new_points, ith_params, 
                                                    sample_function, device)
            # if to many points were sampled, delete the last ones.
            cuted_points = self._cut_tensor_to_length_n(new_better_points)
            sample_points = self._set_sampled_points(sample_points, cuted_points)
        return sample_points

    def _sample_grid(self, current_params, sample_function, n, device):
        new_points = sample_function(n, params=current_params, device=device)
        repeated_params = self._repeat_params(current_params, n)
        new_points = self._apply_filter(new_points.join(repeated_params))
        return new_points

    def _resample_grid(self, new_points, current_params, sample_func, device):
        if len(new_points) == self.n_points:
            # the first grid is already perfect
            return new_points
        elif len(new_points) == 0:
            warnings.warn("""First iteration did not find any valid grid points, for
                             the given filter.
                             Will try again with n = 10 * self.n_points. Or
                             else use only random points!""")
            scaled_n = int(10*self.n_points)
        else:
            scaled_n = int(self.n_points**2/len(new_points))
        new_points = self._sample_grid(current_params, sample_func, scaled_n, device)
        final_points = self._append_random_points(new_points, current_params, device)
        return final_points

    def _append_random_points(self, new_points, current_params, device):
        if len(new_points) == self.n_points:
            return new_points
        random_sampler = RandomUniformSampler(domain=self.domain,
                                              n_points=self.n_points)
        random_sampler.filter_fn = self.filter_fn
        random_points = random_sampler.sample_points(current_params, device=device)
        return new_points | random_points
                            

class ExponentialIntervalSampler(PointSampler):
    """Will sample non equdistant grid points in the given interval.
    This works only on intervals!

    Parameters
    ----------
    domain : torchphysics.domain.Interval
        The Interval in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    exponent : Number
        Determines how non equdistant the points are and at which corner they
        are accumulated. They are computed with a grid in [0, 1]
        and then transformed with the exponent and later scaled/translated:
            exponent < 1: More points at the upper bound. 
                          points = 1 - x**(1/exponent)
            exponent > 1: More points at the lower bound.
                          points = x**(exponent)
    """
    def __init__(self, domain, n_points, exponent):
        assert isinstance(domain, Interval), """The domain has to be a interval!"""
        super().__init__(n_points=n_points)
        self.domain = domain
        self.exponent = exponent

    def sample_points(self, params=Points.empty(), device='cpu'):
        if any(var in self.domain.necessary_variables for var in params.space.keys()):
            return self._sample_params_dependent(self._sample_spaced_grid, params, device)
        return self._sample_params_independent(self._sample_spaced_grid, params, device)

    def _sample_spaced_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        lb = self.domain.lower_bound(params)
        ub = self.domain.upper_bound(params)
        points = torch.linspace(0, 1, len(self)+2, device=device)[1:-1]
        if self.exponent > 1:
            points = points**self.exponent
        else:
            points = 1 - points**(1/self.exponent)
        interval_length = ub - lb
        points = points * interval_length + lb
        return Points(points.reshape(-1, 1), self.domain.space)