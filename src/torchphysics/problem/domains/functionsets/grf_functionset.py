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
        and needs the keyword argument device. 
        Should output random values in the corresponding shape on the given 
        device (cpu, gpu, etc.) as a PyTorch tensor.
        Default is torch.randn, a normal distribution with variance 1.
    normalize : bool, optional
        Normalizes the GRF such that it has mean zero and standard deviation 1.0.
        Default is True.
    sample_noise_in_fourier_space : bool, optional
        If we can sample the noise directly in the fourier space, e.g. the 
        random_generator_fn is called two times and the noise will is constructed
        as the sum of both function calls, where one output is multiplied by the
        imaginary unit. Else we will sample the noise and then transfer the data
        to the fourier space with a fft.
        Default is True.
            
    TODO: Add different return options for the shape of the GRF (e.g. FNO wants an image, 
    DeepOnet rather wants the image reduce into one dimension)
    TODO: Add sampler that samples on the grid which this GRF creates functions on?
    """
    def __init__(self, function_space, resolution, 
                 auto_cov_fn : callable = lambda x : 1/(1 + sum([i**2 for i in x]))**2, 
                 random_generator_fn : callable = torch.randn,
                 normalize : bool = True, 
                 sample_noise_in_fourier_space : bool = True):
        super().__init__(function_space)

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


    def sample_functions(self, n_samples, locations, device="cpu"):
        # since the GRF is defined on a fixed grid the locations inputed
        # are not taken into account...
        self.cov_matrix = self.cov_matrix.to(device)
        sample_shape = (n_samples, ) + self.cov_matrix.shape
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
        
        field = field.unsqueeze(-1)

        # 3) return depending on inputs (copy if multiple locations are provided)
        output_data = Points(field, self.function_space.output_space)
        if isinstance(locations, Points):
            return output_data
        else:
            output_data = Points(field, self.function_space.output_space)
            output = []
            for _ in range(len(locations)):
                output.append(output_data)
            return output