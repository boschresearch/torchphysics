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
        super().__init__(function_space=function_space)
        
        if not isinstance(custom_fn, UserFunction):
            custom_fn = UserFunction(custom_fn)

        self.custom_fn = custom_fn
        self.parameter_sampler = parameter_sampler


    def sample_functions(self, n_samples, locations, device="cpu"):
        self.parameter_sampler.n_points = n_samples
        param_samples = self.parameter_sampler.sample_points(device=device)
        if isinstance(locations, Points):
            return self._sample_custom_fn_at_location(locations, param_samples)
        
        output = []
        for loc in locations:  
            output.append(self._sample_custom_fn_at_location(loc, param_samples))
        return output
    

    def _sample_custom_fn_at_location(self, location : Points, param_samples : Points):
        # If locations are the same over the whole batch, we copy it to match the number 
        # of param_samples.
        # TODO: Not so nice for memory usage, can this be improved???
        if location.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(
                location[self.function_space.input_space].as_tensor, 
                len(param_samples), dim=0
                )
        else:
            location_copy = location[self.function_space.input_space].as_tensor
        params_copy = torch.repeat_interleave(param_samples.as_tensor.unsqueeze(1), 
                                              location.as_tensor.shape[1], dim=1)

        fn_input = torch.cat([location_copy, params_copy], dim=-1)
        fn_output = self.custom_fn(
            Points(fn_input, self.function_space.input_space*param_samples.space)
            )
        return Points(fn_output, self.function_space.output_space)