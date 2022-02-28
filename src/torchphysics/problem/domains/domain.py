import abc
import torch

from ...utils.user_fun import DomainUserFunction, UserFunction
from ..spaces.points import Points


class Domain:
    """The parent class for all built-in domains.

    Parameters
    ----------
    space : torchphysiscs.spaces.Space
        The space in which this object lays.
    dim : int, optional
        The dimension of this domain.
        (if not specified, implicit given through the space)
    """
    def __init__(self, space, dim=None):
        self.space = space
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim
        self._user_volume = None

    def set_necessary_variables(self, *domain_params):
        """Registers the variables/spaces that this domain needs to be 
        properly defined
        """
        self.necessary_variables = set()
        for d_param in domain_params:
            for k in d_param.necessary_args:
                self.necessary_variables.add(k)
        assert not any(var in self.necessary_variables for var in self.space)

    def transform_to_user_functions(self, *domain_params):
        """Transforms all parameters that define a given domain to 
        a UserFunction. This enables that the domain can dependt on other variables.

        Parameters
        ----------
        *domain_params: callables, lists, arrays, tensors or numbers
            The parameters that define a domain.
        """
        out = []
        for d_param in domain_params:
            if not isinstance(d_param, DomainUserFunction):
                d_param = DomainUserFunction(d_param)
            out.append(d_param)
        return tuple(out)

    @property
    def boundary(self):
        """Returns the boundary of this domain. Does not work on
        boundaries itself, e.g. Circle.boundary.boundary throws an error.

        Returns
        -------
        boundary: torchphysics.domains.Boundarydomain
            The boundary-object of the domain.
        """
        raise NotImplementedError

    def set_volume(self, volume):
        """Set the volume of the given domain.

        Parameters
        ----------
        volume : number or callable
            The volume of the domain. Can be a function if the volume changes
            depending on other variables.

        Notes
        -----
        For all basic domains the volume (and surface) are implemented. 
        But if the given domain has a complex shape or is 
        dependent on other variables, the volume can only be approixmated.
        Therefore one can set here a exact expression for the volume, if known. 
        """
        self._user_volume = DomainUserFunction(volume)

    @abc.abstractmethod
    def _get_volume(self, params=Points.empty(), device='cpu'):
        raise NotImplementedError

    def volume(self, params=Points.empty(), device='cpu'):
        """Computes the volume of the current domain.

        Parameters
        ----------
        params : torchphysics.problem.Points, optional
            Additional paramters that are needed to evaluate the domain.

        Returns
        -------
        volume: torch.tensor
            Returns the volume of the domain. If dependent on other parameters,
            the value will be returned as tensor with the shape (len(params), 1).
            Where each row corresponds to the volume of the given values in the
            params row. 
        """
        if self._user_volume is None:
            return self._get_volume(params, device=device)
        else:
            return self._user_volume(params, device=device)

    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        if self.space != other.space:
            raise ValueError("""united domains should lie in the same space.""")
        from .domainoperations.union import UnionDomain
        return UnionDomain(self, other)

    def __sub__(self, other):
        """Creates the cut of domain other from self.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        if self.space != other.space:
            raise ValueError("""complemented domains should lie in the same space.""")
        from .domainoperations.cut import CutDomain
        return CutDomain(self, other)

    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to lie in the same space.
        """
        if self.space != other.space:
            raise ValueError("""Intersected domains should lie in the same space.""")
        from .domainoperations.intersection import IntersectionDomain
        return IntersectionDomain(self, other)

    def __mul__(self, other):
        """Creates the cartesian product of this domain and another domain.

        Parameters
        ----------
        other : Domain
            The other domain to create the cartesian product with.
            Should lie in a disjoint space.
        """
        from .domainoperations.product import ProductDomain
        return ProductDomain(self, other)

    def __contains__(self, points):
        """Checks for every point in points if it lays inside the domain.

        Parameters
        ----------
        points : torchphysics.problem.Points
            A Points object that should be checked.

        Returns
        -------
        torch.Tensor
            A boolean Tensor of the shape (len(points), 1) where every entry contains
            true if the point was inside or false if not.
        """
        return self._contains(points)

    @abc.abstractmethod
    def _contains(self, points, params=Points.empty()):
        raise NotImplementedError

    @abc.abstractmethod
    def bounding_box(self, params=Points.empty(), device='cpu'):
        """Computes the bounds of the domain.

        Returns
        -------
        tensor :
            A torch.Tensor with the length of 2*self.dim.
            It has the form [axis_1_min, axis_1_max, axis_2_min, axis_2_max, ...], 
            where min and max are the minimum and maximum value that the domain
            reaches in each dimension-axis.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        """Creates an equdistant grid in the domain.

        Parameters
        ----------
        n : int, optional
            The number of points that should be created.
        d : float, optional
            The density of points that should be created, if
            n is not defined.
        params : torchphysics.problem.Points, optional
            Additional paramters that are maybe needed to evaluate the domain.
        device : str
            The device on which the points should be created.
            Default is 'cpu'.

        Returns
        -------
        Points :
            A Points object containing the sampled points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        """Creates random uniformly distributed points in the domain.

        Parameters
        ----------
        n : int, optional 
            The number of points that should be created.
        d : float, optional
            The density of points that should be created, if
            n is not defined.
        params : torchphysics.problem.Points, optional
            Additional paramters that are maybe needed to evaluate the domain.
        device : str
            The device on which the points should be created.
            Default is 'cpu'.

        Returns
        -------
        Points :
            A Points object containing the sampled points.
        """
        raise NotImplementedError

    def __call__(self, **data):
        """Evaluates the domain at the given data.
        """
        raise NotImplementedError

    def len_of_params(self, params):
        """Finds the number of params, for which points should be sampled.
        """
        num_of_params = 1
        if len(params) > 0:
            num_of_params = len(params)
        return num_of_params

    def compute_n_from_density(self, d, params):
        """Transforms a given point density to a number of points, since
        all methods from PyTorch only work with a given number.
        """
        volume = self.volume(params)
        if len(volume) > 1:
            raise ValueError(f"""Sampling with a density is only possible for one
                                given pair of parameters. Found {len(volume)} 
                                different pairs. If sampling with a density is needed, 
                                a loop should be used.""")
        n = torch.ceil(d * volume)
        return int(n)

    def _repeat_params(self, n, params):
        repeated_params = Points(torch.repeat_interleave(params, n, dim=0), params.space)
        return 1 if len(repeated_params) else n, repeated_params


class BoundaryDomain(Domain):
    """The parent class for all built-in boundaries.
    Can be used just like the main Domain class.

    Parameters
    ----------
    domain : Domain
        The domain of which this object is the boundary.
    """  
    def __init__(self, domain):
        assert isinstance(domain, Domain)
        super().__init__(space=domain.space, dim=domain.dim-1)
        self.domain = domain
        self.necessary_variables = self.domain.necessary_variables

    def __call__(self, **data):
        evaluated_domain = self.domain(**data)
        return evaluated_domain.boundary

    def bounding_box(self, params=Points.empty(), device='cpu'):
        return self.domain.bounding_box(params)

    @abc.abstractmethod
    def normal(self, points, params=Points.empty(), device='cpu'):
        """Computes the normal vector at each point in points.

        Parameters
        ----------
        points : torch.tensor or torchphysics.problem.Points
            Different points for which the normal vector should be computed.
            The points should lay on the boundary of the domain, to get correct results.
            E.g in 2D: points = Points(torch.tensor([[2, 4], [9, 6], ....]), R2(...))        
        params : dict or torchphysics.problem.Points, optional
            Additional parameters that are maybe needed to evaluate the domain.
        device : str, optional
            The device on which the points should be created.
            Default is 'cpu'.

        Returns
        -------
        torch.tensor
            The tensor is of the shape (len(points), self.dim) and contains the 
            normal vector at each entry from points.
        """
        raise NotImplementedError

    def _transform_input_for_normals(self, points, params, device):
        if not isinstance(points, Points):
            points = Points(points, self.space)
        if not isinstance(params, Points):
            params = Points.from_coordinates(params)
        device = points._t.device
        return points, params, device