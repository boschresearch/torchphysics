import torch
import warnings

from ..domain import BoundaryDomain, Domain
from ..domain0D import Point
from .union import UnionDomain
from ....utils.user_fun import UserFunction
from ...spaces import Points


N_APPROX_VOLUME = 10

class ProductDomain(Domain):
    """
    The 'cartesian' product of two domains. Additionally supports dependence of domain_a
    on domain_b, i.e. if the definition of domain_a contains functions of variables in
    domain_b.space, they are evaluated properly.

    Parameters
    ----------
    domain_a : Domain
        The (optionally dependent) first domain.
    domain_b : Domain
        The second domain.
    """
    def __init__(self, domain_a, domain_b):
        self.domain_a = domain_a
        self.domain_b = domain_b
        if not self.domain_a.space.keys().isdisjoint(self.domain_b.space):
            warnings.warn("""Warning: The space of a ProductDomain will be the product
                of its factor domains spaces. This may lead to unexpected behaviour.""")
        # check dependencies, so that at most domain_a needs variables of domain_b
        self._check_variable_dependencies()
        # set domain params
        space = self.domain_a.space * self.domain_b.space
        super().__init__(space=space, dim=domain_a.dim + domain_b.dim)
        # to set a bounding box
        self.bounds = None
        # necessary variables consist of variables of both domains that are not given in domain_b
        self.necessary_variables \
            = (self.domain_a.necessary_variables - self.domain_b.space.variables) \
              | self.domain_b.necessary_variables

    def _check_variable_dependencies(self):
        a_variables_in_b = any(var in self.domain_b.necessary_variables for
                               var in self.domain_a.space)
        b_variables_in_a = any(var in self.domain_a.necessary_variables for
                               var in self.domain_b.space)
        name_a = self.domain_a.__class__.__name__
        name_b = self.domain_b.__class__.__name__
        if a_variables_in_b and b_variables_in_a:
            raise AssertionError(f"""Both domains {name_a}, {name_b} depend on the 
                                     variables of the other domain. Will not be able 
                                     to resolve order of point creation!""")
        elif a_variables_in_b:
            raise AssertionError(f"""Domain_b: {name_b} depends on the variables of 
                                     domain_a: {name_a}, maybe you meant to use:
                                     domain_b * domain_a (multiplication
                                     is not commutative)""")
        elif b_variables_in_a:
            self._is_constant = False
        else:
            self._is_constant = True

    def __call__(self, **data):
        # evaluate both domains at the given data 
        domain_a = self.domain_a(**data)
        domain_b = self.domain_b(**data)
        # check if the data fixes a variable that would be computed with this domain:
        a_variables_in_data = all(var in data.keys() for var in self.domain_a.space)
        b_variables_in_data = all(var in data.keys() for var in self.domain_b.space)
        if a_variables_in_data: # domain_a will be a fixed point
            point_data = self._create_point_data(self.domain_a.space, data)
            domain_a = Point(space=self.domain_a.space, point=point_data)
        if b_variables_in_data: # domain_b will be a fixed point 
            point_data = self._create_point_data(self.domain_b.space, data)
            domain_b = Point(space=self.domain_a.space, point=point_data)
        return ProductDomain(domain_a=domain_a, domain_b=domain_b)

    def _create_point_data(self, space, data):
        point_data = []
        for vname in space.keys():
            vname_data = data[vname]
            if isinstance(vname_data, (list, tuple, torch.Tensor)):
                point_data.extend(data)
            else: # number
                point_data.append(data)
        return point_data

    @property
    def boundary(self):
        # Domain object of the boundary
        # TODO: implement a seperate class for this for normals etc.
        boundary_1 = ProductDomain(self.domain_a.boundary, self.domain_b)
        boundary_2 = ProductDomain(self.domain_a, self.domain_b.boundary)
        return UnionDomain(boundary_1, boundary_2)

    def _contains(self, points, params=Points.empty()):
        in_a = self.domain_a._contains(points, params)
        in_b = self.domain_b._contains(points, params)
        return torch.logical_and(in_a, in_b)

    def set_bounding_box(self, bounds):
        """To set the bounds of the domain. 

        Parameters
        ----------
        bounds : list
            The bounding box of the domain. Whereby the lenght of the list
            has to be two times the domain dimension. And the bounds need to be in the 
            following order: [min_axis_1, max_axis_1, min_axis_2, max_axis_2, ...]
        """
        assert len(bounds) == 2 * self.dim, """Bounds dont fit the dimension."""
        self.bounds = bounds

    def bounding_box(self, params=Points.empty(), device='cpu'):
        if self.bounds:
            return self.bounds
        elif self._is_constant or self.domain_b.space in params.space:
            # if the domain is constant or additional data for domain a is given
            # we just can create the bounds directly. 
            bounds_a = self.domain_a.bounding_box(params, device=device)
            bounds_b = self.domain_b.bounding_box(params, device=device)
            bounds_a = torch.cat((bounds_a, bounds_b))
        else: # we have to sample some points in b, and approx the bounds.
            warnings.warn(f"""The bounding box of the ProductDomain dependens of the
                              values of domain_b. Therefor will sample
                              {N_APPROX_VOLUME} in domain_b, to compute a 
                              approixmation. If the bounds a known exactly, set 
                              them with .set_bounds().""")
            bounds_b = self.domain_b.bounding_box(params, device=device)
            b_points = self.domain_b.sample_random_uniform(n=N_APPROX_VOLUME,
                                                           params=params)
            _, new_params = self._repeat_params(n=N_APPROX_VOLUME, params=params)
            bounds_a = self.domain_a.bounding_box(b_points.join(new_params), device=device)
            bounds_a = torch.cat((bounds_a, bounds_b))
        return bounds_a
    
    def _get_volume(self, params=Points.empty(), device='cpu'):
        if self._is_constant:
            return self.domain_a.volume(params, device=device) * self.domain_b.volume(params, device=device)
        else:
            warnings.warn(f"""The volume of a ProductDomain where one factor domain depends on the
                              other can only be approximated by evaluating functions at {N_APPROX_VOLUME}
                              points. If you need exact volume or sampling, use domain.set_volume().""")
            # approximate the volume
            n, new_params = self._repeat_params(n=N_APPROX_VOLUME, params=params)
            b_points = self.domain_b.sample_random_uniform(n=n, params=new_params)
            if len(self.domain_b.necessary_variables) > 0:
                # points need to be sampled in every call to this function
                volume_a = self.domain_a.volume(b_points.join(new_params), device=device)
                reshape_volume = volume_a.reshape(N_APPROX_VOLUME, -1)
                mean_volume = torch.sum(reshape_volume, dim=0) / N_APPROX_VOLUME
                return mean_volume.reshape(-1, 1) * self.domain_b.volume(params, device=device)
            elif len(self.necessary_variables) > 0:
                # we can keep the sampled points and evaluate domain_a in a function
                b_volume = self.domain_b.volume(device=device)
                def avg_volume(local_params):
                    _, new_params = self._repeat_params(n=N_APPROX_VOLUME, params=local_params)
                    return torch.sum(self.domain_a.volume(b_points.join(new_params), device=device)\
                        .reshape(N_APPROX_VOLUME,-1), dim=0) / N_APPROX_VOLUME * b_volume
                args = self.domain_a.necessary_variables - self.domain_b.space.variables
                self._user_volume = UserFunction(avg_volume, args=args)
                return avg_volume(params)
            else:
                # we can compute the volume only once and save it
                volume = sum((self.domain_a.volume(b_points, device=device))/N_APPROX_VOLUME \
                    * self.domain_b.volume(device=device))
                self.set_volume(volume)
                return torch.repeat_interleave(volume, max(1, len(params)), dim=0)
            

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        raise NotImplementedError(
            """Grid sampling on a product domain is not implmented. Use a product sampler
               instead.""")
    
    def _sample_uniform_b_points(self, n_in, params=Points.empty(), device='cpu'):
        n_, params = self._repeat_params(n_in, params)
        b_points = self.domain_b.sample_random_uniform(n=n_, params=params,
                                                       device=device)
        volumes = self.domain_a.volume(params.join(b_points), device=device).squeeze(dim=-1)
        if list(volumes.shape) == [1]:
            return n_in, b_points, params
        filter_ = torch.max(volumes)*torch.rand_like(volumes, device=device) < volumes
        b_points = b_points[filter_, ]
        if not params.isempty:
            params = params[filter_, ]
        n_out = len(b_points)
        return n_out, b_points, params

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        if n is not None:
            if self._is_constant:  # we use all sampled b values
                n_, new_params = self._repeat_params(n, params)
                b_points = self.domain_b.sample_random_uniform(n=n_, params=new_params, 
                                                               device=device)
            else:  # use ratio of uniforms to get uniform values in product domain
                n_points, b_points, new_params = \
                    self._sample_uniform_b_points(n, params=params, device=device)
                n_sampled = n
                while n_points != n:
                    if n_points < n:
                        n_guess = int((n/n_points-1)*n_sampled)+1
                        n_out, add_b_points, add_params = \
                            self._sample_uniform_b_points(n_guess, params=params, 
                                                          device=device)
                        b_points = b_points | add_b_points
                        new_params = new_params | add_params
                        n_points += n_out
                    else:
                        b_points = b_points[:n, ]
                        new_params = new_params[:n, ]
                        n_points = n
            a_points = self.domain_a.sample_random_uniform(n=1, params=new_params.join(b_points), 
                                                           device=device)
            return a_points.join(b_points)
        else:
            assert d is not None
            n = int(d*self.volume(device=device))
            return self.sample_random_uniform(n=n, params=params, device=device)