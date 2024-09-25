import torch
from .condition import SingleModuleCondition, SquaredError
from ...models import Parameter


class VariationalPINNCondition(SingleModuleCondition):

    def __init__(self, module, residual_fn, sampler, test_fn_set, track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), name='pinncondition',
                 weight=1.0):
        super().__init__(module, sampler, residual_fn, error_fn=SquaredError(),
                         reduce_fn=torch.mean, name=name, track_gradients=track_gradients,
                         data_functions=data_functions, parameter=parameter, weight=weight)
        self.test_fn_set = test_fn_set


    def _move_static_data(self, device):
        super()._move_static_data(device)
        self.test_fn_set.to(device)


    def forward(self, device='cpu', iteration=None):
        x = self.sampler.sample_points(device=device)
        x_coordinates, x = x.track_coord_gradients()

        y = self.module(x)

        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)

        test_fn = self.test_fn_set(x_coordinates)

        test_space_parameters = {"quad_weights": self.test_fn_set.get_quad_weights()}
        
        unreduced_loss = self.error_fn(self.residual_fn({**y.coordinates,
                                                        **x_coordinates,
                                                        **test_space_parameters, 
                                                        **test_fn.coordinates,
                                                        **data}))
        return self.reduce_fn(unreduced_loss)