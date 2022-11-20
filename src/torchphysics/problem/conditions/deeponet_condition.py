import torch

from .condition import Condition, SquaredError, DataCondition
from ...models import Parameter
from ...utils import UserFunction
from ...models import DeepONet

class DeepONetSingleModuleCondition(Condition):

    def __init__(self, deeponet_model, function_set, input_sampler, residual_fn, 
                 error_fn, reduce_fn=torch.mean,
                 name='singlemodulecondition', track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.net = deeponet_model
        assert isinstance(self.net, DeepONet)
        self.function_set = function_set

        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.input_sampler = input_sampler

        self.residual_fn = UserFunction(residual_fn)
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn
        self.data_functions = self._setup_data_functions(data_functions, self.input_sampler)

        self.eval_function_set = len(
            self.function_set.function_space.output_space.variables & set(self.residual_fn.args)
            ) > 0


    def forward(self, device='cpu', iteration=None):
        # 1) if necessary, sample input function and evaluate branch net
        self.net._forward_branch(self.function_set, iteration_num=iteration, device=device)

        # 2) sample output points
        if self.input_sampler.is_adaptive:
            x = self.input_sampler.sample_points(unreduced_loss=self.last_unreduced_loss,
                                                  device=device)
            self.last_unreduced_loss = None
        else:
            x = self.input_sampler.sample_points(device=device)
        x = x.unsqueeze(0).repeat(len(self.function_set), 1, 1)
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
            function_set_output = self.function_set.create_function_batch(x[0,:,:]).coordinates
        
        unreduced_loss = self.error_fn(self.residual_fn({**y.coordinates,
                                                         **x_coordinates,
                                                         **function_set_output,
                                                         **self.parameter.coordinates,
                                                         **data}))

        if self.input_sampler.is_adaptive:
            self.last_unreduced_loss = unreduced_loss

        return self.reduce_fn(unreduced_loss)


class PIDeepONetCondition(DeepONetSingleModuleCondition):
    """
    A condition that minimizes the mean squared error of the given residual, as 
    required in the framework of physics-informed DeepONets [1].

    Parameters
    -------
    deeponet_model : torchphysics.models.DeepONet
        The DeepONet-model, consisting of trunk and branch net that should be optimized.
    function_set : torchphysics.domains.FunctionSet
        A FunctionSet that provides the different input functions for the branch net.
    input_sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points inside the domain of the residual function,
        could be an inner or a boundary domain.
    residual_fn : callable
        A user-defined function that computes the residual (unreduced loss) from
        inputs and outputs of the model, e.g. by using utils.differentialoperators
        and/or domain.normal
    data_functions : dict
        A dictionary of user-defined functions and their names (as keys). Can be
        used e.g. for right sides in PDEs or functions in boundary conditions.
    track_gradients : bool
        Whether gradients w.r.t. the inputs should be tracked during training or
        not. Defaults to true, since this is needed to compute differential operators
        in PINNs.
    parameter : Parameter
        A Parameter that can be used in the residual_fn and should be learned in
        parallel, e.g. based on data (in an additional DataCondition).
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.

    Notes
    -----
    ..  [1] Wang, Sifan and Wang, Hanwen and Perdikaris,
        "Learning the solution operator of parametric partial
        differential equations with physics-informed DeepOnets", 
        https://arxiv.org/abs/2103.10974, 2021.
    """
    def __init__(self, deeponet_model, function_set, input_sampler, residual_fn, 
                 name='pinncondition', track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), weight=1.0):
        super().__init__(deeponet_model, function_set, input_sampler, 
                         residual_fn=residual_fn, error_fn=SquaredError(), 
                         reduce_fn=torch.mean, name=name, 
                         track_gradients=track_gradients, data_functions=data_functions,
                         parameter=parameter, weight=weight)


class DeepONetDataCondition(DataCondition):
    """
    A condition that fits a single given module to data (handed through a PyTorch
    dataloader).

    Parameters
    ----------
    module : torchphysics.Model
        The torch module which should be fitted to data.
    dataloader : torch.utils.DataLoader
        A PyTorch dataloader which supplies the iterator to load data-target pairs
        from some given dataset. Data and target should be handed as points in input
        or output spaces, i.e. with the correct point object.
    norm : int or 'inf'
        The 'norm' which should be computed for evaluation. If 'inf', maximum norm will
        be used. Else, the result will be taken to the n-th potency (without computing the
        root!)
    constrain_fn : callable, optional
        A additional transformation that will be applied to the network output.
        The function gets as an input all the trunk inputs (e.g. space, time values)
        and the corresponding outputs of the final model (the solution approximation).
        Can be used to enforce some conditions (e.g. boundary values, or scaling the output)
    root : float
        the n-th root to be computed to obtain the final loss. E.g., if norm=2, root=2, the
        loss is the 2-norm.
    use_full_dataset : bool
        Whether to perform single iterations or compute the error on the whole dataset during
        forward call. The latter can especially be useful during validation.
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    """

    def __init__(self, module, dataloader, norm, constrain_fn = None, 
                 root=1., use_full_dataset=False, name='datacondition', weight=1.0):
        super().__init__(module=module, dataloader=dataloader, 
                         norm=norm, root=root, use_full_dataset=use_full_dataset, 
                         name=name, weight=weight, constrain_fn=constrain_fn)
        assert isinstance(self.module, DeepONet)

    def _compute_dist(self, batch, device):
        branch_in, trunk_in, out = batch
        branch_in, trunk_in, out = branch_in.to(device), trunk_in.to(device), \
                                   out.to(device)
        self.module.branch(branch_in)
        model_out = self.module(trunk_in)
        if self.constrain_fn:
            model_out = self.constrain_fn({**model_out.coordinates, **trunk_in.coordinates})
        else:
            model_out = model_out.as_tensor
        return torch.abs(model_out - out.as_tensor)