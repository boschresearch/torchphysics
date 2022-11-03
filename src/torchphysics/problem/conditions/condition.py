import abc

import torch

from ...models import Parameter, AdaptiveWeightLayer
from ...utils import UserFunction
from ..spaces import Points
from ..samplers import StaticSampler, GridSampler, EmptySampler


class SquaredError(torch.nn.Module):
    """
    Implements the sum of squared errors in space dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Computes the squared error of the input.

        Parameters
        ----------
        x : torch.tensor
            The values for which the squared error should be computed.
        """
        return torch.sum(torch.square(x), dim=1)


class Condition(torch.nn.Module):
    """
    A general condition which should be optimized or tracked.

    Parameters
    -------
    name : str
        The name of this condition which will be monitored in logging.
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    track_gradients : bool
        Whether to track input gradients or not. Helps to avoid tracking the
        gradients during validation. If a condition is applied during training,
        the gradients will always be tracked.
    """

    def __init__(self, name=None, weight=1.0, track_gradients=True):
        super().__init__()
        self.name = name
        self.weight = weight
        self.track_gradients = track_gradients

    @abc.abstractmethod
    def forward(self, device='cpu', iteration=None):
        """
        The forward run performed by this condition.

        Returns
        -------
        torch.Tensor : the loss which should be minimized or monitored during training
        """
        raise NotImplementedError

    def _setup_data_functions(self, data_functions, sampler):
        for fun in data_functions:
            data_functions[fun] = UserFunction(data_functions[fun])
        if isinstance(sampler, StaticSampler):
            # functions can be evaluated once
            for fun in data_functions:
                points = sampler.sample_points()
                data_fun_points = data_functions[fun](points)
                #self.register_buffer(fun, data_fun_points)
                data_functions[fun] = UserFunction(data_fun_points)
        return data_functions

    def _move_static_data(self, device):
        pass

class DataCondition(Condition):
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
    root : float
        the n-th root to be computed to obtain the final loss. E.g., if norm=2, root=2, the
        loss is the 2-norm.
    use_full_dataset : bool
        Whether to perform single iterations or compute the error on the whole dataset during
        forward call. The latter can especially be useful during validation.
    name : str
        The name of this condition which will be monitored in logging.
    constrain_fn : callable, optional
        A additional transformation that will be applied to the network output.
        The function can use all the model inputs (e.g. space, time values)
        and the corresponding outputs (the solution approximation).
        Can be used to enforce some conditions (e.g. boundary values, or scaling the output)
    weight : float
        The weight multiplied with the loss of this condition during
        training.
    """

    def __init__(self, module, dataloader, norm, root=1., use_full_dataset=False,
                 name='datacondition', constrain_fn = None,
                 weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=False)
        self.module = module
        self.dataloader = dataloader
        self.norm = norm
        self.root = root
        self.use_full_dataset = use_full_dataset
        self.constrain_fn = constrain_fn
        if self.constrain_fn:
            self.constrain_fn = UserFunction(self.constrain_fn)

    def _compute_dist(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_out = self.module(x)
        if self.constrain_fn:
            model_out = self.constrain_fn({**model_out.coordinates, **x.coordinates})
        else:
            model_out = model_out.as_tensor
        return torch.abs(model_out - y.as_tensor)

    def forward(self, device='cpu', iteration=None):
        if self.use_full_dataset:
            loss = torch.zeros(1, requires_grad=True, device=device)
            for batch in iter(self.dataloader):
                a = self._compute_dist(batch, device)
                if self.norm == 'inf':
                    loss = torch.maximum(loss, torch.max(a))
                else:
                    loss = loss + torch.mean(a**self.norm)/len(self.dataloader)
        else:
            try:
                batch = next(self.iterator)
            except (StopIteration, AttributeError):
                self.iterator = iter(self.dataloader)
                batch = next(self.iterator)
            a = self._compute_dist(batch, device)
            if self.norm == 'inf':
                loss = torch.max(a)
            else:
                loss = torch.mean(a**self.norm)
        if self.root != 1.0:
            loss = loss**(1/self.root)
        return loss


class ParameterCondition(Condition):
    """
    A condition that applies a penalty term on some parameters which are
    optimized during the training process.

    Parameters
    ----------
    parameter : torchphysics.Parameter
        The parameter that should be optimized.
    penalty : callable
        A user-defined function that defines a penalty term on the parameters.
    weight : float
        The weight multiplied with the loss of the penalty during training.
    name : str
        The name of this condition which will be monitored in logging.
    """

    def __init__(self, parameter, penalty, weight, name='parametercondition'):
        super().__init__(name=name, weight=weight, track_gradients=False)
        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.penalty = UserFunction(penalty)

    def forward(self, device='cpu', iteration=None):
        return self.penalty(self.parameter.coordinates)


class SingleModuleCondition(Condition):
    """A condition that minimizes the reduced loss of a single module.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
        could be an inner or a boundary domain.
    residual_fn : callable
        A user-defined function that computes the residual (unreduced loss) from
        inputs and outputs of the model, e.g. by using utils.differentialoperators
        and/or domain.normal
    error_fn : callable
        Function that will be applied to the output of the residual_fn to compute
        the unreduced loss. Should reduce only along the 2nd (i.e. space-)axis.
    reduce_fn : callable
        Function that will be applied to reduce the loss to a scalar. Defaults to
        torch.mean
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
    """

    def __init__(self, module, sampler, residual_fn, error_fn, reduce_fn=torch.mean,
                 name='singlemodulecondition', track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.module = module
        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.sampler = sampler
        self.residual_fn = UserFunction(residual_fn)
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn
        self.data_functions = self._setup_data_functions(data_functions, sampler)

        if self.sampler.is_adaptive:
            self.last_unreduced_loss = None

    def forward(self, device='cpu', iteration=None):
        if self.sampler.is_adaptive:
            x = self.sampler.sample_points(unreduced_loss=self.last_unreduced_loss,
                                           device=device)
            self.last_unreduced_loss = None
        else:
            x = self.sampler.sample_points(device=device)

        x_coordinates, x = x.track_coord_gradients()

        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)

        y = self.module(x)

        unreduced_loss = self.error_fn(self.residual_fn({**y.coordinates,
                                                         **x_coordinates,
                                                         **self.parameter.coordinates,
                                                         **data}))

        if self.sampler.is_adaptive:
            self.last_unreduced_loss = unreduced_loss

        return self.reduce_fn(unreduced_loss)

    def _move_static_data(self, device):
        if self.sampler.is_static:
            for fn in self.data_functions:
                self.data_functions[fn].fun = self.data_functions[fn].fun.to(device)


class MeanCondition(SingleModuleCondition):
    """
    A condition that minimizes the mean of the residual of a single module, can be
    used e.g. in Deep Ritz Method [1] or for energy functionals, since the mean can
    be seen as a (scaled) integral approximation.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
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
    ..  [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017
    """

    def __init__(self, module, sampler, residual_fn, track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), name='meancondition',
                 weight=1.0):
        super().__init__(module, sampler, residual_fn, error_fn=torch.nn.Identity(),
                         reduce_fn=torch.mean, name=name, track_gradients=track_gradients,
                         data_functions=data_functions, parameter=parameter, weight=weight)


class DeepRitzCondition(MeanCondition):
    """
    Alias for :class:`MeanCondition`.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
        could be an inner or a boundary domain.
    integrand_fn : callable
        The integrand of the weak formulation of the differential equation.
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
    ..  [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017
    """
    def __init__(self, module, sampler, integrand_fn, track_gradients=True, data_functions={},
                 parameter=Parameter.empty(), name='deepritzcondition', weight=1.0):
        super().__init__(module, sampler, integrand_fn, track_gradients=track_gradients,
                         data_functions=data_functions, parameter=parameter, name=name,
                         weight=weight)


class PINNCondition(SingleModuleCondition):
    """
    A condition that minimizes the mean squared error of the given residual, as required in
    the framework of physics-informed neural networks [1].

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
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
    ..  [1] M. Raissi, "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial differential
        equations", Journal of Computational Physics, vol. 378, pp. 686-707, 2019.
    """

    def __init__(self, module, sampler, residual_fn, track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), name='pinncondition',
                 weight=1.0):
        super().__init__(module, sampler, residual_fn, error_fn=SquaredError(),
                         reduce_fn=torch.mean, name=name, track_gradients=track_gradients,
                         data_functions=data_functions, parameter=parameter, weight=weight)


class PeriodicCondition(Condition):
    """
    A condition that allows to learn dependencies between points at the ends of a given
    Interval. Can be used e.g. for a variety of periodic boundary conditions.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    periodic_interval : torchphysics.domains.Interval
        The interval on which' boundary the periodic (boundary) condition will be set.
    non_periodic_sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points for the axis that are not defined via the
        periodic_interval
    residual_fn : callable
        A user-defined function that computes the residual (unreduced loss) from
        inputs and outputs of the model, e.g. by using utils.differentialoperators
        and/or domain.normal. Instead of the name of the axis of the periodic interval,
        it takes {name}_left and {name}_right as an input. The same holds for all outputs
        of the network and the results of the data_functions.
    error_fn : callable
        Function that will be applied to the output of the residual_fn to compute
        the unreduced loss. Should reduce only along the 2nd (i.e. space-)axis.
    reduce_fn : callable
        Function that will be applied to reduce the loss to a scalar. Defaults to
        torch.mean
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
    """

    def __init__(self, module, periodic_interval, residual_fn,
                 non_periodic_sampler=EmptySampler(), error_fn=SquaredError(),
                 reduce_fn=torch.mean, name='periodiccondition', track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.module = module
        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.periodic_interval = periodic_interval
        self.non_periodic_sampler = non_periodic_sampler
        self.residual_fn = UserFunction(residual_fn)
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn

        n_points = max(len(self.non_periodic_sampler), 1)
        self.left_sampler = GridSampler(self.periodic_interval.boundary_left,
                                        n_points=n_points).make_static()
        self.right_sampler = GridSampler(self.periodic_interval.boundary_right,
                                         n_points=n_points).make_static()

        tmp_left_sampler = self.left_sampler*self.non_periodic_sampler
        tmp_right_sampler = self.right_sampler*self.non_periodic_sampler
        if self.non_periodic_sampler.is_static:
            tmp_left_sampler = tmp_left_sampler.make_static()
            tmp_right_sampler = tmp_right_sampler.make_static()
        self.left_data_functions = self._setup_data_functions(data_functions,
                                                              tmp_left_sampler)
        self.right_data_functions = self._setup_data_functions(data_functions,
                                                               tmp_right_sampler)

        if self.non_periodic_sampler.is_adaptive:
            self.last_unreduced_loss = None

    def forward(self, device='cpu', iteration=None):
        if self.non_periodic_sampler.is_adaptive:
            x_b = self.non_periodic_sampler.sample_points(
                unreduced_loss=self.last_unreduced_loss,
                device=device)
            self.last_unreduced_loss = None
        else:
            x_b = self.non_periodic_sampler.sample_points(device=device)
        
        x_left = self.left_sampler.sample_points(device=device)
        x_right = self.right_sampler.sample_points(device=device)

        x_left_coordinates, x_left = x_left.track_coord_gradients()
        x_right_coordinates, x_right = x_right.track_coord_gradients()
        x_b_coordinates, x_b = x_b.track_coord_gradients()


        data_left = {}
        data_right = {}
        for fun in self.left_data_functions:
            data_left[fun] = self.left_data_functions[fun]({**x_left_coordinates,
                                                            **x_b_coordinates})
        data_left = {f'{k}_left': data_left[k] for k in data_left}
        for fun in self.right_data_functions:
            data_right[fun] = self.right_data_functions[fun]({**x_right_coordinates,
                                                              **x_b_coordinates})
        data_right = {f'{k}_right': data_right[k] for k in data_right}

        y_left = self.module(x_left.join(x_b))
        y_right = self.module(x_right.join(x_b))

        y_left_coordinates = y_left.coordinates
        y_left_coordinates = {f'{k}_left': y_left_coordinates[k] for k in y_left_coordinates}
        y_right_coordinates = y_right.coordinates
        y_right_coordinates = {f'{k}_right': y_right_coordinates[k] for k in y_right_coordinates}


        x_left_coordinates = {f'{k}_left': x_left_coordinates[k] for k in x_left_coordinates}
        x_right_coordinates = {f'{k}_right': x_right_coordinates[k] for k in x_right_coordinates}


        unreduced_loss = self.error_fn(self.residual_fn({**y_left_coordinates,
                                                         **y_right_coordinates,
                                                         **x_left_coordinates,
                                                         **x_right_coordinates,
                                                         **x_b_coordinates,
                                                         **self.parameter.coordinates,
                                                         **data_right,
                                                         **data_left}))

        if self.non_periodic_sampler.is_adaptive:
            self.last_unreduced_loss = unreduced_loss

        return self.reduce_fn(unreduced_loss)

    def _move_static_data(self, device):
        if self.non_periodic_sampler.is_static:
            for fn in self.left_data_functions:
                self.left_data_functions[fn].fun = \
                    self.left_data_functions[fn].fun.to(device)
            for fn in self.right_data_functions:
                self.right_data_functions[fn].fun = \
                    self.right_data_functions[fn].fun.to(device)


class IntegroPINNCondition(Condition):
    """
    A condition that also allows to include the computation of integrals or convolutions
    by sampling a second set of points by an additional sampler.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the usual set of points.
    integral_sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points that can be used to approximate an integral.
    residual_fn : callable
        A user-defined function that computes the residual (unreduced loss) from
        inputs and outputs of the model, e.g. by using utils.differentialoperators
        and/or domain.normal. The point set used to approximate the integral  and the
        output of the model at these points are given as input {name}_integral
    error_fn : callable
        Function that will be applied to the output of the residual_fn to compute
        the unreduced loss. Should reduce only along the 2nd (i.e. space-)axis.
    reduce_fn : callable
        Function that will be applied to reduce the loss to a scalar. Defaults to
        torch.mean
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
    """

    def __init__(self, module, sampler, residual_fn,
                 integral_sampler, error_fn=SquaredError(),
                 reduce_fn=torch.mean, name='periodiccondition', track_gradients=True,
                 data_functions={}, parameter=Parameter.empty(), weight=1.0):
        super().__init__(name=name, weight=weight, track_gradients=track_gradients)
        self.module = module
        self.parameter = parameter
        self.register_parameter(name + '_params', self.parameter.as_tensor)
        self.residual_fn = UserFunction(residual_fn)
        self.error_fn = error_fn
        self.reduce_fn = reduce_fn

        self.sampler = sampler
        self.integral_sampler = integral_sampler

        self.data_functions = self._setup_data_functions(data_functions,
                                                         self.sampler)

        if self.sampler.is_adaptive:
            self.last_unreduced_loss = None

    def forward(self, device='cpu', iteration=None):
        if self.sampler.is_adaptive:
            x = self.sampler.sample_points(
                unreduced_loss=self.last_unreduced_loss,
                device=device)
            self.last_unreduced_loss = None
        else:
            x = self.sampler.sample_points(device=device)
        x_int = self.integral_sampler.sample_points(device=device)

        n_x = len(x)
        n_x_int = len(x_int)

        x = x.unsqueeze(dim=1)
        x_int = x_int.unsqueeze(dim=0)
        x_coordinates, x = x.track_coord_gradients()
        x_int_coordinates, x_int = x_int.track_coord_gradients()

        # combine both inputs to be able to compute model(x_int) with all correct
        # parameters
        x_combined = x.repeat(1, n_x_int)
        x_combined[..., list(x_int.space.keys())] = x_int.repeat(n_x, 1)

        data = {}
        for fun in self.data_functions:
            data[fun] = self.data_functions[fun](x_coordinates)

        y = self.module(x)
        y_int = self.module(x_combined)

        y_int_coordinates = y_int.coordinates
        y_int_coordinates = {f'{k}_integral': y_int_coordinates[k] for k in y_int_coordinates}

        x_int_coordinates = {f'{k}_integral': x_int_coordinates[k] for k in x_int_coordinates}

        unreduced_loss = self.error_fn(self.residual_fn({**y.coordinates,
                                                         **y_int_coordinates,
                                                         **x_coordinates,
                                                         **x_int_coordinates,
                                                         **self.parameter.coordinates,
                                                         **data}))

        if self.sampler.is_adaptive:
            self.last_unreduced_loss = unreduced_loss

        return self.reduce_fn(unreduced_loss)

    def _move_static_data(self, device):
        if self.sampler.is_static:
            for fn in self.data_functions:
                self.data_functions[fn].fun = self.data_functions[fn].fun.to(device)


class AdaptiveWeightsCondition(SingleModuleCondition):
    """
    A condition using an AdaptiveWeightLayer [1] to assign adaptive weights to all points
    during training.

    Parameters
    -------
    module : torchphysics.Model
        The torch module which should be optimized.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points in the domain of the residual function,
        could be an inner or a boundary domain.
    residual_fn : callable
        A user-defined function that computes the residual (unreduced loss) from
        inputs and outputs of the model, e.g. by using utils.differentialoperators
        and/or domain.normal
    error_fn : callable
        Function that will be applied to the output of the residual_fn to compute
        the unreduced loss (shape [n_points]). The result will be multiplied by the
        adaptive weights.
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
    ..  [1] Levi D. McClenny, "Self-Adaptive Physics-Informed Neural Networks using a
        Soft Attention Mechanism", CoRR, 2020
    """

    def __init__(self, module, sampler, residual_fn, error_fn=SquaredError(),
                 track_gradients=True, data_functions={}, parameter=Parameter.empty(),
                 name='adaptive_w_condition', weight=1.0):

        if not sampler.is_static:
            raise ValueError("Adaptive point weights should only be used with static",
                             "samplers.")

        adaptive_layer = AdaptiveWeightLayer(len(sampler))

        def adaptive_reduce_fun(x):
            return torch.mean(adaptive_layer(x))

        super().__init__(module, sampler, residual_fn, error_fn=error_fn,
                         reduce_fn=adaptive_reduce_fun, name=name, track_gradients=track_gradients,
                         data_functions=data_functions, parameter=parameter, weight=weight)

        self.adaptive_layer = adaptive_layer
