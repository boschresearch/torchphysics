import torch

from ..domain import Domain
from ...spaces import Points


class Translate(Domain):
    """Class that translates a given domain by a given vector (or vector function).
    
    Parameters
    ----------
    domain : torchphysics.domain.Domain
        The domain that should be translated.
    translation : array_like or callable
        The vector that describes the translation, can also be a function that 
        returns different vectors.

    Notes
    -----
    All domains can already be moved by passing in a function as the needed domain
    parameter. But for complex domains (cut, etc.) or objects with many corners
    (cube, square) it is easier to just translate the whole domain with this class.
    """ 
    def __init__(self, domain : Domain, translation):
        self.domain = domain
        self.translate_fn = self.transform_to_user_functions(translation)[0]
        super().__init__(self.domain.space, self.domain.dim)
        self.set_necessary_variables(self.translate_fn)
        self.necessary_variables.update(self.domain.necessary_variables)

    def __call__(self, **data):
        new_domain = self.domain(**data)
        new_translate_fn = self.translate_fn.partially_evaluate(**data)
        return Translate(domain=new_domain, translation=new_translate_fn)

    def _contains(self, points, params=Points.empty()):
        translate_values = self.translate_fn(points.join(params)).reshape(-1, self.space.dim)
        shifted_points = points[:, list(self.space.keys())].as_tensor \
                        - translate_values
        #points[:, list(self.space.keys())] = Points(shifted_points, self.space)
        return self.domain._contains(Points(shifted_points, self.space), params)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        original_points = self.domain.sample_random_uniform(n=n, d=d, params=params, 
                                                            device=device).as_tensor
        n = int(len(original_points) / (len(params) + 1))
        _, params = self._repeat_params(n + 1, params) # round up n
        translate_values = self.translate_fn(params).squeeze(-1)
        translated_points = original_points + translate_values
        return Points(translated_points, self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        original_points = self.domain.sample_grid(n=n, d=d, params=params, 
                                                            device=device).as_tensor
        translated_points = self._translate_points(original_points, params)
        return Points(translated_points, self.space)

    def _translate_points(self, points, params):
        # if original domain depends on params, then they are already copied
        if any(var in self.domain.necessary_variables for var in params.space):
            n = int(len(points) / max(len(params), 1))
            _, params = self._repeat_params(n, params)
        # else we have to copy both the params and original output points
        else:
            n, n_params = len(points), max(len(params), 1)
            _, params = self._repeat_params(n, params)
            points = points.repeat(n_params, 1)
        translate_values = self.translate_fn(params).squeeze(-1)
        points += translate_values
        return points

    def volume(self, params=Points.empty(), device='cpu'):
        return self.domain.volume(params=params, device=device)

    def set_volume(self, volume):
        return self.domain.set_volume(volume)

    def bounding_box(self, params=Points.empty(), device='cpu'):
        domain_bounds = self.domain.bounding_box(params=params, device=device)
        translation_values = self.translate_fn(params).reshape(-1, self.space.dim)
        translation_values = torch.repeat_interleave(translation_values, 2, 1)
        # domain_bounds are in shape [x_min, x_max, y_min, y_max, ...]
        # both min and max have to be shifted by the same value
        new_bounds = domain_bounds + translation_values
        return new_bounds

    @property
    def boundary(self):
        return Translate(self.domain.boundary, self.translate_fn)