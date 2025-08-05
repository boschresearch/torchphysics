import torch

from .condition import Condition, SquaredError
from ...utils import UserFunction
from ...models.deeponet.deeponet import DeepONet

class OperatorCondition(Condition):
    """
    General condition used for the (data-driven) training of different 
    operator approaches.

    Parameters
    ----------
    module : torchphysics.Model
        The torch module which should be fitted to data.
    input_function_sampler : torch.utils.FunctionSampler
        The sampler providing the input data to the module.
    output_function_sampler : torch.utils.FunctionSampler
        The expected output to a given input. 
    residual_fn : callable, optional
        An optional function that computes the residual, by default 
        the network output minus the expected output is taken. 
    relative : bool, optional
        Whether to compute the relative error (i.e. error / target) or absolute error.
        Default is True, hence, the relative error is used.
    error_fn : callable, optional
        the function used to compute the final loss. E.g., the squarred error or 
        any other norm.
    reduce_fn : callable, optional
        Function that will be applied to reduce the loss to a scalar. Defaults to
        torch.mean
    name : str, optional
        The name of this condition which will be monitored in logging.
    weight : float, optional
        The weight multiplied with the loss of this condition during
        training.
    epsilon : float, optional
        For the relative loss, we add a small epsilon to the target to 
        circumvent dividing by 0, the default is 1.e-8.  
    """
    def __init__(
            self, 
            module, 
            input_function_sampler,
            output_function_sampler, 
            residual_fn=None, 
            relative=True,
            reduce_fn=torch.mean,
            error_fn=SquaredError(),
            name="operator_condition",
            weight=1.0,
            epsilon=1e-8
        ):
        super().__init__(name=name, weight=weight, track_gradients=False)
        assert input_function_sampler.function_set.is_discretized, \
            "This condition needs discretized function sets"
        assert output_function_sampler.function_set.is_discretized, \
            "This condition needs discretized function sets"
        
        self.module = module
        self.input_function_sampler = input_function_sampler
        self.output_function_sampler = output_function_sampler
        
        if residual_fn:
            self.residual_fn = UserFunction(residual_fn)
        else:
            self.residual_fn = None

        self.relative = relative
        self.epsilon = epsilon

        self.error_fn = error_fn
        self.reduce_fn = reduce_fn

        self.module_is_deeponet = isinstance(self.module, DeepONet)

    def forward(self, device="cpu", iteration=None):
        input_functions = self.input_function_sampler.sample_functions(device=device)
        output_functions = self.output_function_sampler.sample_functions(device=device)
        
        if self.module_is_deeponet:
            model_out = self.module(None, input_functions)
        else:
            model_out = self.module(input_functions)
        
        if self.residual_fn:
            first_error = self.residual_fn(
                {**output_functions.coordinates, 
                 **model_out.coordinates, 
                 **input_functions.coordinates}
            )
        else:
            first_error = model_out.as_tensor - output_functions.as_tensor
        
        out = self.error_fn(first_error)
        
        if self.relative:
            y_norm = self.error_fn(output_functions.as_tensor) + self.epsilon
            out /= y_norm

        return self.reduce_fn(out)


class PIOperatorCondition(Condition):

    def __init__(
            self, 
            module, 
            input_function_sampler,
            residual_fn,
            reduce_fn=torch.mean,
            error_fn=SquaredError(),
            name="pi_operator_condition",
            weight=1.0,
        ):
        super().__init__(name=name, weight=weight, track_gradients=False)
        assert input_function_sampler.function_set.is_discretized, \
            "This condition needs discretized function sets"
        
        self.module = module
        self.input_function_sampler = input_function_sampler
        
        if residual_fn:
            self.residual_fn = UserFunction(residual_fn)
        else:
            self.residual_fn = None

        self.error_fn = error_fn
        self.reduce_fn = reduce_fn

        self.module_is_deeponet = isinstance(self.module, DeepONet)

    def forward(self, device="cpu", iteration=None):
        input_functions = self.input_function_sampler.sample_functions(device=device)
        
        if self.module_is_deeponet:
            model_out = self.module(None, input_functions)
        else:
            model_out = self.module(input_functions)
        
        first_error = self.residual_fn(
            {**model_out.coordinates, 
                **input_functions.coordinates}
        )

        return self.reduce_fn(self.error_fn(first_error))