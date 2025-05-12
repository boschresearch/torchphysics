import torch

from .functionset import FunctionSet

from ....utils.user_fun import UserFunction
from ...samplers import PointSampler
from ...spaces import Points


class CustomFunctionSet(FunctionSet):
    """FunctionSet for an arbitrary function.

    Parameters
    ----------
    parameter_sampler : torchphysics.samplers.PointSampler
        A sampler that provides additional parameters that can be used
        to create different kinds of functions. E.g. our FunctionSet consists
        of Functions like k*x, x is the input variable and k is given through
        the sampler.

        During each training iteration will call the parameter_sampler to sample
        new parameters. For each parameter a function will be created and the
        input batch of functions will be of the same length as the sampled
        parameters.
    custom_fn : callable
        A function that describes the FunctionSet. The input of the functions
        can include the variables of the function_space.input_space and the
        parameters from the parameter_sampler.
    """

    def __init__(self, function_space, parameter_sampler : PointSampler, custom_fn):
        super().__init__(function_space=function_space, function_set_size=parameter_sampler.n_points)
        
        if not isinstance(custom_fn, UserFunction):
            custom_fn = UserFunction(custom_fn)

        self.custom_fn = custom_fn
        self.parameter_sampler = parameter_sampler

    def create_functions(self, device="cpu"):
        self.param_samples = self.parameter_sampler.sample_points(device=device)
    
    def get_function(self, idx):
        if isinstance(idx, int): idx = [idx]
        self.current_idx = idx
        return self._evaluate_fn_at_locations

    def _evaluate_fn_at_locations(self, locations : Points):
        # TODO: Not so nice for memory usage, can this be improved???
        location_copy = self._transform_locations(locations)

        params_copy = torch.repeat_interleave(
                self.param_samples.as_tensor[self.current_idx].unsqueeze(1), 
                location_copy.shape[1], dim=1
            )

        fn_input = torch.cat([location_copy, params_copy], dim=-1)
        fn_output = self.custom_fn(
            Points(fn_input, self.function_space.input_space*self.param_samples.space)
            )
        return Points(fn_output, self.function_space.output_space)
