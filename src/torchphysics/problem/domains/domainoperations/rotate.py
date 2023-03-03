import torch

from ..domain import Domain, BoundaryDomain
from ...spaces import Points
from ....utils.user_fun import DomainUserFunction


class RotationMatrix2D(DomainUserFunction):
    """Given a function :math:`f:\\Omega \\to R` will create the two dimensional 
    rotation matrix :math:`(cos(f), -sin(f); sin(f), cos(f))`.
    """
    def __call__(self, args={}, device='cpu'):
        angle_values = super().__call__(args, device).reshape(-1, 1)
        matrix_row = torch.cat((torch.cos(angle_values), -torch.sin(angle_values)), dim=1)
        matrix_row_2 = torch.cat((-matrix_row[:, 1:], matrix_row[:, :1]), dim=1)
        return torch.stack((matrix_row, matrix_row_2), dim=1)

    def partially_evaluate(self, **args):
        new_angle_fn = super().partially_evaluate(**args)
        return RotationMatrix2D(new_angle_fn)


class RotationMatrix3D(DomainUserFunction):

    def __init__(self, alpha, beta, gamma, defaults=..., args=...):
        super().__init__(alpha, defaults, args)
        self.beta = DomainUserFunction(beta)
        self.gamma = DomainUserFunction(gamma)


    @property
    def necessary_args(self):
        alpha_args = super().necessary_args
        beta_args = self.beta.necessary_args
        gamma_args = self.gamma.necessary_args
        return list(set(alpha_args + beta_args + gamma_args))


class Rotate(Domain):
    """Class that rotates a given domain via a given matrix.
    
    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain that should be rotated.
    rotation_matrix : array_like or callable
        The matrix that describes the rotation, can also be a function that 
        returns different matrices, depending on other parameters.
    rotate_around : array_like or callable, optional
        The point around which the rotation occurs, can also be a function.
        Default is the origin.

    Notes
    -----
    All domains can already be rotated by passing in a function as the needed domain
    parameter. But for complex domains (cut, etc.) or objects with many corners
    (cube, square) it is easier to just rotate the whole domain with this class.
    """ 
    def __init__(self, domain : Domain, rotation_matrix, rotate_around=None):
        if isinstance(domain, BoundaryDomain):
            assert domain.dim >= 1, "Can only rotate domains in dimensions >= 2"
        else:
            assert domain.dim > 1, "Can only rotate domains in dimensions >= 2"
        if rotate_around is None:
            rotate_around = torch.zeros((1, domain.dim))
        self.domain = domain
        self.rotation_fn, self.rotate_around = \
            self.transform_to_user_functions(rotation_matrix, rotate_around)
        super().__init__(self.domain.space, self.domain.dim)
        self.set_necessary_variables(self.rotation_fn)
        self.necessary_variables.update(self.domain.necessary_variables)

    @classmethod
    def from_angles(cls, domain : Domain, *angles, rotate_around=None):
        """Creates the rotation from given angles.

        Parameters
        ----------
        domain : torchphysics.domain.Domain
            The domain that should be rotated.
        *angles : float or callable
            The angles that describe the rotation, can also be a functions.
            In 2D one angle :math:`\\alpha` is needed and internally the 
            rotation matrix 
            :math:`(\\cos(\\alpha), -\\sin(\\alpha); \\sin(\\alpha), \\cos(\\alpha))`
            is constructed.
            For 3D three angles are needed and the euler (extrinsic) rotation 
            matrix from https://en.wikipedia.org/wiki/Rotation_matrix is used.
        rotate_around : array_like or callable, optional
            The point around which the rotation occurs, can also be a function.
            Default is the origin.        
        """
        assert domain.dim <= 3, \
            "Rotation matrix for dimension > 3 is not known, please create it yourself" \
            + " and use the basic constructor."
        if domain.dim == 2:
            assert len(angles) == 1, "In 2D one rotation angle is needed!"
            rotation_matrix = RotationMatrix2D(angles[0])
        else:
            assert len(angles) == 3, "In 3D three rotation angles are needed!"
            raise NotImplementedError
        return cls(domain, rotation_matrix=rotation_matrix, rotate_around=rotate_around)

    def __call__(self, **data):
        new_domain = self.domain(**data)
        new_rotation_matrix = self.rotation_fn.partially_evaluate(**data)
        new_rotate_around = self.rotate_around.partially_evaluate(**data)
        return Rotate(domain=new_domain, rotation_matrix=new_rotation_matrix, 
                      rotate_around=new_rotate_around)

    def volume(self, params=Points.empty(), device='cpu'):
        return self.domain.volume(params=params, device=device)

    def set_volume(self, volume):
        return self.domain.set_volume(volume)

    @property
    def boundary(self):
        return Rotate(self.domain.boundary, self.rotation_fn, self.rotate_around)

    def _contains(self, points, params=Points.empty()):
        translate_values = self.rotate_around(points.join(params)).reshape(-1, self.space.dim)
        rotation_matrix = self.rotation_fn(points.join(params)).reshape(-1, self.space.dim, 
                                                                        self.space.dim)
        shifted_points = points[:, list(self.space.keys())].as_tensor \
                        - translate_values
        # here apply inverse rotation -> solve: Matrix * x = shifted_points
        rotated_points = torch.linalg.solve(rotation_matrix, shifted_points.unsqueeze(-1))
        shifted_points = rotated_points.squeeze(-1) + translate_values
        return self.domain._contains(Points(shifted_points, self.space), params)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        original_points = self.domain.sample_random_uniform(n=n, d=d, params=params, 
                                                            device=device).as_tensor
        n = int(len(original_points) / (len(params) + 1))
        _, params = self._repeat_params(n + 1, params) # round up n
        rotated_points = self._rotate_points(params, original_points)
        return Points(rotated_points, self.space)

    def _rotate_points(self, params, original_points):
        translate_values = self.rotate_around(params).reshape(-1, self.space.dim)
        rotation_matrix = self.rotation_fn(params).reshape(-1, self.space.dim, self.space.dim)
        translated_points = original_points - translate_values
        rotated_points = torch.matmul(rotation_matrix, translated_points.unsqueeze(-1))
        translated_points = rotated_points.squeeze(-1) + translate_values
        return translated_points

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        original_points = self.domain.sample_grid(n=n, d=d, params=params, 
                                                  device=device).as_tensor
        return self._rotate_grid(original_points, params)

    def _rotate_grid(self, points, params):
        # if original domain depends on params, then they are already copied
        if any(var in self.domain.necessary_variables for var in params.space):
            n = int(len(points) / max(len(params), 1))
            _, params = self._repeat_params(n, params)
        # else we have to copy both the params and original output points
        else:
            n, n_params = len(points), max(len(params), 1)
            _, params = self._repeat_params(n, params)
            points = points.repeat(n_params, 1)
        rotated_points = self._rotate_points(params, points)
        return Points(rotated_points, self.space)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        domain_bounds = self.domain.bounding_box(params=params, device=device)
        translate_values = self.rotate_around(params).reshape(-1, self.space.dim)
        rotation_matrix = self.rotation_fn(params).reshape(-1, self.space.dim, 
                                                           self.space.dim)
        translation_values = torch.repeat_interleave(translate_values, 2, 1)
        # domain_bounds are in shape [x_min, x_max, y_min, y_max, ...]
        # both min and max have to be shifted by the same value
        domain_bounds = domain_bounds - translation_values
        rotated_min = torch.matmul(rotation_matrix, domain_bounds[:, ::2].unsqueeze(-1))
        rotated_min = rotated_min.squeeze(-1)
        rotated_max = torch.matmul(rotation_matrix, domain_bounds[:, 1::2].unsqueeze(-1))
        rotated_max = rotated_max.squeeze(-1)
        domain_bounds = torch.zeros((len(rotated_min), 2*self.space.dim), 
                                    device=device)
        domain_bounds[:, ::2] = torch.min(rotated_min, rotated_max) 
        domain_bounds[:, 1::2] = torch.max(rotated_min, rotated_max) 
        domain_bounds = domain_bounds + translation_values
        return domain_bounds.squeeze(0)