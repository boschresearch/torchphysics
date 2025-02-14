import torch
import math

from .functionset import FunctionSet

from ...spaces import Points

from ..domain1D import Interval


class HarmonicFunctionSet1D(FunctionSet):

    def __init__(self, function_space, interval : Interval, max_frequence : int, 
                 random_sample_fn = torch.randn):
        super().__init__(function_space)
        self.max_frequence = max_frequence
        self.interval_len = interval.upper_bound.fun - interval.lower_bound.fun
        self.random_sample_fn = random_sample_fn


    def sample_functions(self, n_samples, locations, device="cpu"):
        fourier_coefficients = self.random_sample_fn((n_samples, self.max_frequence, 2), device=device)
        fourier_coefficient_0 = self.random_sample_fn(n_samples, device=device)
        if isinstance(locations, Points):
            return self._eval_basis_at_locaction(locations, fourier_coefficients, fourier_coefficient_0)
        
        output = []
        for loc in locations:  
            output.append(
                self._eval_basis_at_locaction(loc, fourier_coefficients, fourier_coefficient_0))
        return output
    
    
    def _eval_basis_at_locaction(self, location, fourier_coefficients, fourier_coefficient_0):
        if location.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(location[self.function_space.input_space].as_tensor, 
                                                    len(fourier_coefficients), dim=0)
        else:
            location_copy = location[self.function_space.input_space].as_tensor
        output = torch.zeros((len(fourier_coefficients), location_copy.shape[1], 1))
        pi_scale = 2 * math.pi / self.interval_len
        for i in range(self.max_frequence):
            output[:, :, 0] += \
                fourier_coefficients[:, i:i+1, 0] * torch.sin(pi_scale * i * location_copy[:, :, 0]) + \
                fourier_coefficients[:, i:i+1, 1] * torch.cos(pi_scale * i * location_copy[:, :, 0])
        output[:, :, 0] += fourier_coefficient_0.unsqueeze(1)
        return Points(output, self.function_space.output_space)