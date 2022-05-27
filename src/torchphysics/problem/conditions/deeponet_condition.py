import torch

from .condition import Condition, SquaredError
from ...models import Parameter
from ...utils import UserFunction
from ...models import DeepONet

class DeepONetSingleModuleCondition(Condition):

    def __init__(self, deeponet_model, function_set, output_sampler, residual_fn, error_fn, reduce_fn=torch.mean,
                 name='singlemodulecondition', track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.net = deeponet_model
        assert isinstance(self.net, DeepONet)
        self.function_set = function_set

        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.output_sampler = output_sampler

        self.residual_fn = UserFunction(residual_fn)
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn
        self.data_functions = self._setup_data_functions(data_functions, self.output_sampler)

        self.eval_function_set = len(
            self.function_set.function_space.output_space.variables & set(self.residual_fn.args)
            ) > 0


    def forward(self, device='cpu', iteration=None):
        # 1) if necessary, sample input function and evaluate branch net
        self.net._forward_branch(self.function_set, iteration_num=iteration, device=device)

        # 2) sample output points
        if self.output_sampler.is_adaptive:
            x = self.output_sampler.sample_points(unreduced_loss=self.last_unreduced_loss,
                                                  device=device)
            self.last_unreduced_loss = None
        else:
            x = self.output_sampler.sample_points(device=device)
        x_coordinates, x = x.track_coord_gradients()

        # 3) evaluate model (only trunk net)
        y = self.net(x, device=device)

        # 4) evaluate condition
        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)
        # now check whether evaluation of function set in output points is necessary, i.e
        # whether the functions are part of the loss
        function_set_output = {}
        if self.eval_function_set:
            function_set_output = self.function_set.create_function_batch(x).coordinates
        # Problem: function set nimmt sowohl x als auch t als Input, also wird an versch. Orten und in versch. Variablen ausgewertet
        
        unreduced_loss = self.error_fn(self.residual_fn({**y.coordinates,
                                                         **x_coordinates,
                                                         #**input_points_coords,
                                                         **function_set_output,
                                                         **self.parameter.coordinates,
                                                         **data}))

        if self.output_sampler.is_adaptive:
            self.last_unreduced_loss = unreduced_loss

        return self.reduce_fn(unreduced_loss)


class PIDeepONetCondition(DeepONetSingleModuleCondition):

    def __init__(self, deeponet_model, function_set, output_sampler, residual_fn, 
                 name='singlemodulecondition', track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), weight=1.0):
        super().__init__(deeponet_model, function_set, output_sampler, 
                         residual_fn=residual_fn, error_fn=SquaredError(), 
                         reduce_fn=torch.mean, name=name, 
                         track_gradients=track_gradients, data_functions=data_functions,
                         parameter=parameter, weight=weight)