import torch
from itertools import product

from .functionset import FunctionSet
from ...spaces import Points


class GRFFunctionSet(FunctionSet):
    """A functionset that can create gaussian random fields.

    Parameters
    ----------
    resolution : int, tuple, list
        The resolution of the gausian random field. For higher dimensional
        fields a tuple or list must be passed in setting the resoultion
        for each dimension. Each resolution needs to be even.
    auto_cov_fn : callable, optional
        The function describing the correlation between the points in the
        gaussian random field.
        Is evaluated in the fourier space at the frequencies 
        (k_i - resolution_i/2) for i = 1, …, dimension 
        and k_i = 0,…, resolution_i.
        Default is a power law given by
            lambda x : 1/(1 + sum([i**2 for i in x]))**2
    random_generator_fn : callable, optional
        A function that creates the underlying random variables. 
        As an input obtains a tuple of the shape 
            (number of functions, resolution_1, …, resolution_n)
        and also the keyword argument device. 
        Should output random values in the corresponding shape on the given 
        device (cpu, gpu, etc.) as a PyTorch tensor.
        Default is torch.randn, a normal distribution with variance 1.
    normalize : bool, optional
        Normalizes the GRF such that it has mean zero and standard deviation 1.0.
        Default is True.
    sample_noise_in_fourier_space : bool, optional
        If we can sample the noise directly in the fourier space, so the 
        random_generator_fn is called two times and the noise is constructed
        as the sum of both function calls, where one output is multiplied by the
        imaginary unit. Else we will sample the noise and then transfer the data
        to the fourier space with a fft.
        Default is True.
    flatten : bool, optional
        If the output should be flattened to a 1D tensor along all intermediate dimensions.
        Default is False.
    """
    def __init__(self, function_space, function_set_size, resolution, 
                 auto_cov_fn : callable = lambda x : 1/(1 + sum([i**2 for i in x]))**2, 
                 random_generator_fn : callable = torch.randn,
                 normalize : bool = True, 
                 sample_noise_in_fourier_space : bool = True, 
                 flatten : bool = False):
        super().__init__(function_space, function_set_size)

        if isinstance(resolution, int): resolution = [resolution]
        assert self.function_space.input_space.dim == len(resolution), \
            "Resolution shape does not match input space dimension."
            
        self.cov_matrix = torch.zeros(resolution)
        for x in product(*(range(-r//2, r//2) for r in resolution)):
            y = [x[i] + resolution[i]//2 for i in range(len(x))]
            self.cov_matrix[tuple(y)] = auto_cov_fn(x)

        self.normalize = normalize
        self.shift_idx = [i+1 for i in range(len(resolution))]
        self.random_gen_fn = random_generator_fn
        self.sample_in_fourier_space = sample_noise_in_fourier_space
        self.flatten = flatten

    @property
    def is_discretized(self):
        return True

    def create_functions(self, device="cpu"):
        self.cov_matrix = self.cov_matrix.to(device)
        sample_shape = (self.function_set_size, ) + self.cov_matrix.shape
        # 1) sample the underlying noise
        if self.sample_in_fourier_space:
            noise = self.random_gen_fn(sample_shape, device=device)
            noise2 = self.random_gen_fn(sample_shape, device=device)
            noise = torch.complex(noise, noise2)
        else:
            noise = self.random_gen_fn(sample_shape, device=device)
            noise = torch.fft.fftn(noise, dim=len(sample_shape)-1)
            noise = torch.fft.fftshift(noise, dim=self.shift_idx)
        
        # 2) multiply by cov matrix and go back from fourier space
        field = torch.fft.ifftshift(self.cov_matrix * noise, dim=self.shift_idx)
        field = torch.fft.ifftn(field, dim=self.shift_idx).real

        if self.normalize:
            mean_values = torch.mean(field, dim=self.shift_idx, keepdim=True)
            field -= mean_values
            std_values = torch.std(field, dim=self.shift_idx, keepdim=True)
            field /= std_values
        
        if self.flatten:
            field = field.view(self.function_set_size, -1)

        self.grf = field.unsqueeze(-1)

    def get_function(self, idx):
        if isinstance(idx, int): idx = [idx]
        self.current_idx = idx
        return Points(self.grf[self.current_idx], self.function_space.output_space)