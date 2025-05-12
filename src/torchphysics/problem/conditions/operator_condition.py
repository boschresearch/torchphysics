import torch

from .condition import Condition, SquaredError
from ...utils import UserFunction
from ...models.deeponet.deeponet import DeepONet

class OperatorCondition(Condition):

    def __init__(
            self, 
            module, 
            input_function_sampler,
            output_function_sampler, 
            residual_fn=None, 
            reduce_fn=torch.mean,
            error_fn=SquaredError(),
            name="operator_condition",
            weight=1.0,
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
        
        return self.reduce_fn(self.error_fn(first_error))



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