"""File with samplers that create random distributed points.
"""
import torch
import numbers

from .sampler_base import PointSampler, AdaptiveSampler
from ..domains.domain import BoundaryDomain
from ..spaces import Points


class RandomUniformSampler(PointSampler):
    """Will sample random uniform distributed points in the given domain.

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desiered density of the created points.
    filter : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return False. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.
    """
    def __init__(self, domain, n_points=None, density=None, filter_fn=None):
        super().__init__(n_points=n_points, density=density, filter_fn=filter_fn)
        self.domain = domain

    def _sample_points(self, params=Points.empty(), device='cpu'):
        if self.n_points:
            rand_points = self.domain.sample_random_uniform(self.n_points,
                                                            params=params, 
                                                            device=device)
            repeated_params = self._repeat_params(params, len(self))
            return rand_points.join(repeated_params)
        else: # density is used
            sample_function = self.domain.sample_random_uniform
            if any(var in self.domain.necessary_variables for \
                    var in params.space.keys()):
                return self._sample_params_dependent(sample_function, params, device)
            return self._sample_params_independent(sample_function, params, device)

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
        sample_function = self.domain.sample_random_uniform
        num_of_params = max(1, len(params))
        sample_points = None
        for i in range(num_of_params):
            new_sample_points = None
            num_of_new_points = 0
            iterations = 0
            # we have to make sure to sample for each param exactly n points
            while num_of_new_points < self.n_points:
                # sample points
                new_points = self._sample_for_ith_param(sample_function, params,
                                                        i, device)
                # apply filter and save valid points
                new_points = self._apply_filter(new_points)
                num_of_new_points += len(new_points)
                new_sample_points = self._set_sampled_points(new_sample_points,
                                                             new_points)
                iterations += 1
                self._check_iteration_number(iterations, num_of_new_points)
            # if to many points were sampled, delete them.
            cuted_points = self._cut_tensor_to_length_n(new_sample_points)
            sample_points = self._set_sampled_points(sample_points, cuted_points)
        return sample_points 


class GaussianSampler(PointSampler):
    """Will sample normal/gaussian distributed points in the given domain.
    Only works for the inner part of a domain, not the boundary!

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 
    mean : list, array or tensor
        The center/mean of the distribution. Has to fit the dimension
        of the given domain.
    std : number
        The standard deviation of the distribution.
    """
    def __init__(self, domain, n_points, mean, std):
        assert not isinstance(domain, BoundaryDomain), \
            """Gaussian sampling is not implemented for boundaries."""
        super().__init__(n_points=n_points)
        self.domain = domain
        self.mean = mean
        self.std = torch.tensor(std)
        self._check_mean_correct_dim()

    def _check_mean_correct_dim(self):
        if isinstance(self.mean, numbers.Number):
            self.mean = torch.FloatTensor([self.mean])
        elif not isinstance(self.mean, torch.Tensor):
            self.mean = torch.FloatTensor(self.mean)
        assert len(self.mean) == self.domain.dim, \
            f"""Dimension of mean: {self.mean}, does not fit the domain.""" 

    def _sample_points(self, params=Points.empty(), device='cpu'):
        self._set_device_of_mean_and_std(device)
        num_of_params = max(1, len(params))
        sample_points = None
        torch_dis = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)
        for i in range(num_of_params):
            current_num_of_points = 0
            new_sample_points = None
            ith_params = params[i, ] if len(params) > 0 else Points.empty()
            repeat_params = self._repeat_params(ith_params, len(self))
            while current_num_of_points < self.n_points:
                new_points = torch_dis.sample((self.n_points,))
                new_points = Points(new_points, self.domain.space)
                new_points = new_points.join(repeat_params)
                new_points = self._check_inside_domain(new_points)
                current_num_of_points += len(new_points)
                new_sample_points = self._set_sampled_points(new_sample_points,
                                                             new_points)
            # if to many points were sampled, delete them.
            cuted_points = self._cut_tensor_to_length_n(new_sample_points)
            sample_points = self._set_sampled_points(sample_points, cuted_points)
        return sample_points

    def _set_device_of_mean_and_std(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def _check_inside_domain(self, new_points):
        inside = self.domain._contains(new_points)
        index = torch.where(inside)[0]
        return new_points[index, ]


class LHSSampler(PointSampler):
    """Will create a simple latin hypercube sampling [1] in the given domain.
    Only works for the inner part of a domain, not the boundary!

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    n_points : int
        The number of points that should be sampled. 

    Notes
    -----
    A bounding box is used tp create the lhs-points in the domain.
    Points outside will be rejected and additional random uniform points will be 
    added to get a total number of n_points.
    ..  [1] https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    """
    def __init__(self, domain, n_points):
        assert not isinstance(domain, BoundaryDomain), \
            """LHS sampling is not implemented for boundaries."""
        super().__init__(n_points=n_points)
        self.domain = domain

    def _sample_points(self, params=Points.empty(), device='cpu'):
        num_of_params = max(1, len(params))
        sample_points = None
        for i in range(num_of_params):
            ith_params = params[i, ] if len(params) > 0 else Points.empty()
            bounding_box = self.domain.bounding_box(ith_params, device=device)
            lhs_in_box = self._create_lhs_in_bounding_box(bounding_box, device)
            new_points = self._check_lhs_inside(lhs_in_box, ith_params)
            final_points = self._append_random_points(new_points, ith_params)
            sample_points = self._set_sampled_points(sample_points, final_points)
        return sample_points

    def _create_lhs_in_bounding_box(self, bounding_box, device):
        lhs_points = torch.zeros((self.n_points, self.domain.dim), device=device)
        # for each axis apply the lhs strategy
        for i in range(self.domain.dim):
            axis_grid = torch.linspace(bounding_box[2*i], bounding_box[2*i+1], 
                                       steps=self.n_points+1, device=device)[:-1] # dont need endpoint
            axis_length = bounding_box[2*i+1] - bounding_box[2*i]
            random_shift = axis_length/self.n_points * torch.rand(self.n_points,
                                                                  device=device)
            axis_points = torch.add(axis_grid, random_shift)
            # change order of points, to get 'lhs-grid' at the end
            permutation = torch.randperm(self.n_points)
            lhs_points[:, i] = axis_points[permutation]
        return lhs_points

    def _check_lhs_inside(self, lhs_points, ith_params):
        new_points = Points(lhs_points, self.domain.space)
        repeat_params = self._repeat_params(ith_params, len(new_points))
        new_points = new_points.join(repeat_params)
        inside = self.domain._contains(new_points)
        index = torch.where(inside)[0]
        return new_points[index, ]

    def _append_random_points(self, new_points, current_params):
        if len(new_points) == self.n_points:
            return new_points
        random_sampler = RandomUniformSampler(domain=self.domain,
                                              n_points=self.n_points-len(new_points))
        random_points = random_sampler.sample_points(current_params)
        return new_points | random_points

class AdaptiveThresholdRejectionSampler(AdaptiveSampler):
    """
    An adaptive sampler that creates more points in regions with high loss.
    During sampling, points with loss larger than ´min(loss)+resample_ratio*(max(loss)-min(loss))´
    are kept for the next iteration, while points with small loss are regarded and resampled
    (random) uniformly in the whole domain.

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    resample_ratio : float
        During sampling, points with loss larger than ´min(loss)+resample_ratio*(max(loss)-min(loss))´
        are kept for the next iteration, while points with small loss are regarded and resampled
        (random) uniformly in the whole domain.
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desired initial (and average) density of the created points, actual
        density will change loccally during iterations.
    filter : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return False. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.

    """
    def __init__(self, domain, resample_ratio, n_points=None, density=None,
                 filter_fn=None):
        super().__init__(n_points=n_points, density=density, filter_fn=filter_fn)
        self.domain = domain
        self.resample_ratio = resample_ratio
        self.random_sampler = RandomUniformSampler(domain,
            n_points=n_points,
            density=density,
            filter_fn=filter_fn)
        
        self.last_points = None

    def sample_points(self, unreduced_loss=None, params=Points.empty(), device='cpu'):
        new_points = self.random_sampler.sample_points(params, device=device)
        if self.last_points is None or unreduced_loss is None:
            self.last_points = new_points
        else:
            max_l, min_l = torch.max(unreduced_loss), torch.min(unreduced_loss)
            filter_tensor = unreduced_loss < min_l + (max_l-min_l)*self.resample_ratio
            self.last_points._t[filter_tensor,:] = new_points._t[filter_tensor,:]
        return self.last_points


class AdaptiveRandomRejectionSampler(AdaptiveSampler):
    """
    An adaptive sampler that creates more points in regions with high loss.
    During sampling, points with high loss are more likely to be kept for the
    next iteration, while points with small loss are regarded and resampled (random)
    uniformly in the whole domain.

    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain in which the points should be sampled.
    n_points : int, optional
        The number of points that should be sampled.
    density : float, optional
        The desired initial (and average) density of the created points, actual
        density will change loccally during iterations.
    filter : callable, optional
        A function that restricts the possible positions of sample points.
        A point that is allowed should return True, therefore a point that should be 
        removed must return False. The filter has to be able to work with a batch
        of inputs.
        The Sampler will use a rejection sampling to find the right amount of points.

    """
    def __init__(self, domain, n_points=None, density=None,
                 filter_fn=None):
        super().__init__(n_points=n_points, density=density, filter_fn=filter_fn)
        self.domain = domain
        self.random_sampler = RandomUniformSampler(domain,
            n_points=n_points,
            density=density,
            filter_fn=filter_fn)
        
        self.last_points = None

    def sample_points(self, unreduced_loss=None, params=Points.empty(), device='cpu'):
        new_points = self.random_sampler.sample_points(params, device=device)
        if self.last_points is None or unreduced_loss is None:
            self.last_points = new_points
        else:
            max_l, min_l = torch.max(unreduced_loss), torch.min(unreduced_loss)
            filter_tensor = unreduced_loss < min_l + (max_l-min_l)*torch.rand_like(unreduced_loss)
            self.last_points._t[filter_tensor,:] = new_points._t[filter_tensor,:]
        return self.last_points
