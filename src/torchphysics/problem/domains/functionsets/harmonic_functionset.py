import torch
import math

from .functionset import FunctionSet

from ...spaces import Points

class HarmonicFunctionSet1D(FunctionSet):
    """ A function set that creates harmonic functions in 1D.
    The functions are of the form
        .. math:: \sum_{i=0}^{N} a_i \sin(\frac{2\pi i x}{L}) + b_i \cos(\frac{2\pi i x}{L})
    where N is the maximum frequence, L is the period length and 
    a_i, b_i are the fourier coefficients, which are created randomly.

    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace
        The function space of the functions in the set.
    function_set_size : int
        The number of functions in the set. This sets how many a_i, b_i are created 
        at once.
    period : float  
        The length of the underyling interval.
    max_frequence : int
        The maximum frequence of the functions that are created.
    random_sample_fn : callable, optional
        A function that creates random samples to initialize the fourier coefficients.
        Default is torch.randn.
    """
    def __init__(self, function_space, function_set_size, 
                 period, max_frequence, 
                 random_sample_fn = torch.randn):
        super().__init__(function_space, function_set_size)
        self.max_frequence = max_frequence
        self.period_len = period
        self.random_sample_fn = random_sample_fn

    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
                (self.function_set_size, self.max_frequence+1, 2), device=device
            )

    def get_function(self, idx):
        if isinstance(idx, int): idx = [idx]
        self.current_idx = idx
        return self._eval_basis_at_locaction
    
    
    def _eval_basis_at_locaction(self, location : Points):
        location_copy = self._transform_locations(location)

        output = torch.zeros((len(self.current_idx), location_copy.shape[1], 1), 
                            device=location.as_tensor.device)

        pi_scale = 2 * math.pi / self.period_len
        for i in range(self.max_frequence+1):
            output[:, :, 0] += \
                self.fourier_coefficients[self.current_idx, i:i+1, 0] * torch.sin(pi_scale * i * location_copy[:, :, 0]) + \
                self.fourier_coefficients[self.current_idx, i:i+1, 1] * torch.cos(pi_scale * i * location_copy[:, :, 0])

        return Points(output, self.function_space.output_space)
    

class HarmonicFunctionSet2D(HarmonicFunctionSet1D):
    """ A function set that creates harmonic functions in the given dimension.
    The functions are build from a fourier basis in the given space, 
    see also https://en.wikipedia.org/wiki/Multidimensional_transform for the
    mathematical background.

    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace
        The function space of the functions in the set.
    function_set_size : int
        The number of functions in the set. This sets how many a_i, b_i are created 
        at once.
    period : list or tuple
        The length of the underyling domain in each space direction.
    max_frequence : list or tuple
        The maximum frequence of the functions in each space direction.
    random_sample_fn : callable, optional
        A function that creates random samples to initialize the fourier coefficients.
        Default is torch.randn.
    """
    def __init__(self, function_space, function_set_size, 
                 period, max_frequence, 
                 random_sample_fn = torch.randn):
        assert isinstance(period, (list, tuple))
        assert isinstance(max_frequence, (list, tuple))
        super().__init__(function_space, function_set_size, period, max_frequence, random_sample_fn)


    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
            (self.function_set_size, self.max_frequence[0]+1, self.max_frequence[1]+1, 4), device=device
            )
    
    
    def _eval_basis_at_locaction(self, location : Points):
        location_copy = self._transform_locations(location)

        shape = [len(self.current_idx)]
        shape.extend(location_copy.shape[1:-1])
        shape.append(1)
        output = torch.zeros(shape, device=location.as_tensor.device)

        pi_scale_x = 2 * math.pi / self.period_len[0]
        pi_scale_y = 2 * math.pi / self.period_len[1]

        cast_tensor = [1] * len(location_copy.shape)
        cast_tensor[0] = -1

        for i in range(self.max_frequence[0]+1):
            sin_x = torch.sin(pi_scale_x * i * location_copy[..., 0:1])
            cos_x = torch.cos(pi_scale_x * i * location_copy[..., 0:1])

            for j in range(self.max_frequence[1]+1):
                sin_y = torch.sin(pi_scale_y * j * location_copy[..., 1:2])
                cos_y = torch.cos(pi_scale_y * j * location_copy[..., 1:2])

                output[..., 0:1] += \
                    self.fourier_coefficients[self.current_idx, i, j, 0].view(cast_tensor) * sin_x * sin_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 1].view(cast_tensor) * cos_x * sin_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 2].view(cast_tensor) * sin_x * cos_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 3].view(cast_tensor) * cos_x * cos_y
                
        return Points(output, self.function_space.output_space)
    


class HarmonicFunctionSet3D(HarmonicFunctionSet2D):

    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
            (self.function_set_size, 
             self.max_frequence[0]+1, 
             self.max_frequence[1]+1, 
             self.max_frequence[2]+1,
             8), device=device
            )
    
    def _eval_basis_at_locaction(self, location : Points):
        location_copy = self._transform_locations(location)

        shape = [len(self.current_idx)]
        shape.extend(location_copy.shape[1:-1])
        shape.append(1)
        output = torch.zeros(shape, device=location.as_tensor.device)

        pi_scale_x = 2 * math.pi / self.period_len[0]
        pi_scale_y = 2 * math.pi / self.period_len[1]
        pi_scale_z = 2 * math.pi / self.period_len[1]

        cast_tensor = [1] * len(location_copy.shape)
        cast_tensor[0] = -1

        for i in range(self.max_frequence[0]+1):
            sin_x = torch.sin(pi_scale_x * i * location_copy[..., 0:1])
            cos_x = torch.cos(pi_scale_x * i * location_copy[..., 0:1])

            for j in range(self.max_frequence[1]+1):
                sin_y = torch.sin(pi_scale_y * j * location_copy[..., 1:2])
                cos_y = torch.cos(pi_scale_y * j * location_copy[..., 1:2])

                for k in range(self.max_frequence[2]+1):
                    sin_z = torch.sin(pi_scale_z * k * location_copy[..., 2:3])
                    cos_z = torch.cos(pi_scale_z * k * location_copy[..., 2:3])


                    output[..., 0:1] += \
                        self.fourier_coefficients[self.current_idx, i, j, k, 0].view(cast_tensor) * sin_x * sin_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 1].view(cast_tensor) * sin_x * sin_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 2].view(cast_tensor) * sin_x * cos_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 3].view(cast_tensor) * sin_x * cos_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 4].view(cast_tensor) * cos_x * sin_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 5].view(cast_tensor) * cos_x * sin_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 6].view(cast_tensor) * cos_x * cos_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, k, 7].view(cast_tensor) * cos_x * cos_y * sin_z
                
        return Points(output, self.function_space.output_space)