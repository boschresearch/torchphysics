"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc

import torch
import numpy as np

from . import datacreator as dc
from ..utils import (normal_derivative,
                     prepare_user_fun_input,
                     apply_to_batch,
                     is_batch)


class Condition(torch.nn.Module):
    """
    A Condition that should be fulfilled by the DE solution.

    Conditions can be applied to the boundary or inner part of the DE domain and are a
    central concept in this library. Their solution can either be enforced during
    training or tracked during validation.

    Parameters
    ----------
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used for the computation of the conditioning loss/metric.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    track_gradients : bool or list of str or list of DiffVariables
        Whether the gradients w.r.t. the inputs should be tracked.
        Tracking can be necessary for training of a PDE.
        If True, all gradients will be tracked.
        If a list of strings or variables is passed, gradient is tracked
        only for the variables in the list.
        If False, no gradients will be tracked.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, name, norm, weight=1.0,
                 track_gradients=True,
                 data_plot_variables=True):
        super().__init__()
        self.name = name
        self.norm = norm
        self.weight = weight
        self.track_gradients = track_gradients
        self.data_plot_variables = data_plot_variables

        # variables are registered when the condition is added to a problem or variable
        self.setting = None

    @abc.abstractmethod
    def get_data(self):
        """Creates and returns the data for the given condition."""
        return

    @abc.abstractmethod
    def get_data_plot_variables(self):
        return

    def is_registered(self):
        return self.setting is not None

    def serialize(self):
        dct = {}
        dct['name'] = self.name
        dct['norm'] = self.norm.__class__.__name__
        dct['weight'] = self.weight
        return dct


class DiffEqCondition(Condition):
    """
    A condition that enforces the solution of a Differential Equation in the
    inner part of a domain.

    Parameters
    ----------
    pde : function handle
        A method that takes the output and input of a model and computes its deviation
        from some (partial) differential equation. See utils.differentialoperators for
        useful helper functions.
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from a PDE.
    name : str
        name of this condition (should be unique per problem or variable)
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
        If an int is given, the methode will use at least as many data points as the
        number. The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers.
    track_gradients : bool
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, pde, norm, name='pde', data_fun=None,
                 sampling_strategy='random', weight=1.0,
                 dataset_size=10000, track_gradients=True,
                 data_plot_variables=False):
        super().__init__(name, norm, weight,
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        self.pde = pde
        self.data_fun = data_fun
        self.datacreator = dc.InnerDataCreator(variables=None,
                                               dataset_size=dataset_size,
                                               sampling_strategy=sampling_strategy)

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.variables})
        inp = prepare_user_fun_input(self.pde,
                                     {'u': u,
                                      **data,
                                      **self.setting.parameters})
        err = self.pde(**inp)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        if self.is_registered():
            self.datacreator.variables = self.setting.variables
            inp_data = self.datacreator.get_data()
            if self.data_fun is None:
                return inp_data
            else:
                inp_data, data = apply_data_fun(self.data_fun,
                                                inp_data,
                                                whole_batch=self.data_fun_whole_batch,
                                                batch_size=self.dataset_len)
                return {**inp_data,
                        'data': data}
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['sampling_strategy'] = self.datacreator.sampling_strategy
        dct['pde'] = self.pde.__name__
        dct['dataset_size'] = self.datacreator.dataset_size
        return dct

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.setting.variables
        elif self.data_plot_variables is False:
            return None
        else:
            return self.data_plot_variables


class DataCondition(Condition):
    """
    A condition that enforces the model to fit a given dataset.

    Parameters
    ----------
    data_x : dict
        A dictionary containing pairs of variables and data for that variables,
        organized in numpy arrays or torch tensors of equal length.
    data_u : array-like
        The targeted solution values for the data points in data_x.
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    """

    def __init__(self, data_x, data_u, name, norm,
                 weight=1.0):
        super().__init__(name, norm, weight,
                         track_gradients=False,
                         data_plot_variables=False)
        self.data_x = data_x
        self.data_u = data_u

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.variables})
        return self.norm(u, data['target'])

    def get_data(self):
        if self.is_registered():
            return {**self.data_x,
                    'target': self.data_u}
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        return super().serialize()

    def get_data_plot_variables(self):
        return None


class BoundaryCondition(Condition):
    """
    Parent class for all boundary conditions.

    Parameters
    ----------
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    track_gradients : bool
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, name, norm, track_gradients, weight=1.0,
                 data_plot_variables=True):
        super().__init__(name, norm, weight=weight,
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        # boundary_variable is registered when the condition is added to that variable
        self.boundary_variable = None  # string

    def serialize(self):
        dct = super().serialize()
        dct['boundary_variable'] = self.boundary_variable
        return dct

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.boundary_variable
        elif self.data_plot_variables is False:
            return None
        else:
            return self.data_plot_variables


class DirichletCondition(BoundaryCondition):
    """
    Implementation of a Dirichlet boundary condition based on a function handle.

    Parameters
    ----------
    dirichlet_fun : function handle
        A method that takes boundary points (in the usual dictionary form) as an input
        and returns the desired boundary values at those points.
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points for this
        condition. See domains for more details.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
        If an int is given, the methode will use at least as many data points as the
        number. The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, dirichlet_fun, name, norm, whole_batch=True,
                 sampling_strategy='random', boundary_sampling_strategy='random',
                 weight=1.0, dataset_size=10000,
                 data_plot_variables=True):
        super().__init__(name, norm, weight=weight,
                         track_gradients=False,
                         data_plot_variables=data_plot_variables)
        self.dirichlet_fun = dirichlet_fun
        self.whole_batch = whole_batch
        self.dataset_size = dataset_size
        self.dataset_len = get_data_len(dataset_size)
        self.datacreator = dc.BoundaryDataCreator(variables=None,
                                                  dataset_size=dataset_size,
                                                  sampling_strategy=sampling_strategy,
                                                  boundary_sampling_strategy=boundary_sampling_strategy)

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.variables})
        return self.norm(u, data['target'])

    def get_data(self):
        if self.is_registered():
            self.datacreator.variables = self.setting.variables
            self.datacreator.boundary_variable = self.boundary_variable
            inp_data = self.datacreator.get_data()
            inp_data, target = apply_data_fun(self.dirichlet_fun,
                                              inp_data,
                                              whole_batch=self.whole_batch,
                                              batch_size=self.dataset_len)
            return {**inp_data,
                    'target': target}
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['dirichlet_fun'] = self.dirichlet_fun.__name__
        dct['dataset_size'] = self.datacreator.dataset_size
        dct['sampling_strategy'] = self.datacreator.sampling_strategy
        dct['boundary_sampling_strategy'] = self.datacreator.boundary_sampling_strategy
        return dct


class NeumannCondition(BoundaryCondition):
    """
    Implementation of a Neumann boundary condition based on a function handle.

    Parameters
    ----------
    neumann_fun : function handle
        A method that takes boundary points (in the usual dictionary form) as an input
        and returns the desired values of the normal derivatives of the model.
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points for this
        condition. See domains for more details.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
        If an int is given, the methode will use at least as many data points as the
        number. The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, neumann_fun, name, norm, whole_batch=True,
                 sampling_strategy='random', boundary_sampling_strategy='random',
                 weight=1.0, dataset_size=10000,
                 data_plot_variables=True):
        super().__init__(name, norm, weight=weight,
                         track_gradients=True,
                         data_plot_variables=data_plot_variables)
        self.neumann_fun = neumann_fun
        self.whole_batch = whole_batch
        self.dataset_size = dataset_size
        self.dataset_len = get_data_len(dataset_size)
        self.datacreator = dc.BoundaryDataCreator(variables=None,
                                                  dataset_size=dataset_size,
                                                  sampling_strategy=sampling_strategy,
                                                  boundary_sampling_strategy=boundary_sampling_strategy)

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.variables})
        normal_derivatives = normal_derivative(u, data[self.boundary_variable],
                                               data['normal'])
        return self.norm(normal_derivatives, data['target'])

    def get_data(self):
        if self.is_registered():
            self.datacreator.variables = self.setting.variables
            self.datacreator.boundary_variable = self.boundary_variable
            inp_data = self.datacreator.get_data()
            normals = self.setting.variables[self.boundary_variable] \
                .domain.boundary_normal(inp_data[self.boundary_variable])
            inp_data, target = apply_data_fun(self.neumann_fun,
                                              inp_data,
                                              whole_batch=self.whole_batch,
                                              batch_size=self.dataset_len)
            return {**inp_data,
                    'target': target,
                    'normal': normals}
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['neumann_fun'] = self.neumann_fun.__name__
        dct['dataset_size'] = self.datacreator.dataset_size
        dct['sampling_strategy'] = self.datacreator.sampling_strategy
        dct['boundary_sampling_strategy'] = self.datacreator.boundary_sampling_strategy
        return dct


class DiffEqBoundaryCondition(BoundaryCondition):
    """
    Implementation a arbitrary boundary condition based on a function handle.

    Parameters
    ----------
    bound_condition_fun : function handle
        A method that takes the output and input (in the usual dictionary form)
        of a model, the boundary normals and additional data (given through
        data_fun, and
        only when needed) as an input. The method then computes and returns 
        the desired boundary condition.
    name : str
        name of this condition (should be unique per problem or variable)
    norm : torch.nn.Module
        A Pytorch module which forward pass returns the scalar norm of the difference of
        two input tensors, and is therefore similar to the implementation of nn.MSELoss.
        The norm is used to compute the loss for the deviation of the model from the
        given data.
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points for this
        condition. See domains for more details.
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
    dataset_size : int, list, tuple or dic
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
        If an int is given, the methode will use at least as many data points as the
        number. The number of desired points can also be uniquely picked for each
        variable, if a list, tuple or dic is given as an input. Then the whole number
        of data points will be the product of the given numbers.
    data_fun : function handle
        A method that represents the right-hand side of the boundary condition. As
        an input it takes the boundary points in the usual dictionary form.
        If the right-hand side is independent of the model, it is more efficient to
        compute the values only once and save them.
        If the right-hand side dependents on the model outputs, or is zero, this
        parameter should be None and the whole condition has to be implemented in
        bound_condition_fun.
    data_plot_variables : bool or tuple
        The variables which are used to log the used training data in a scatter plot.
        If False, no plots are created. If True, behaviour is defined in each condition.
    """

    def __init__(self, bound_condition_fun, name, norm,
                 sampling_strategy='random', boundary_sampling_strategy='random',
                 weight=1.0, dataset_size=10000, data_fun=None,
                 data_fun_whole_batch=True, data_plot_variables=True):
        super().__init__(name, norm, weight=weight,
                         track_gradients=True,
                         data_plot_variables=data_plot_variables)
        self.bound_condition_fun = bound_condition_fun
        self.data_fun = data_fun
        self.data_fun_whole_batch = data_fun_whole_batch
        self.dataset_size = dataset_size
        self.dataset_len = get_data_len(dataset_size)
        self.datacreator = dc.BoundaryDataCreator(variables=None,
                                                  dataset_size=dataset_size,
                                                  sampling_strategy=sampling_strategy,
                                                  boundary_sampling_strategy=boundary_sampling_strategy)

    def forward(self, model, data):
        u = model({v: data[v] for v in self.setting.variables})
        inp = prepare_user_fun_input(self.bound_condition_fun,
                                     {'u': u,
                                      **data,
                                      **self.setting.parameters})
        err = self.bound_condition_fun(**inp)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        if self.is_registered():
            self.datacreator.variables = self.setting.variables
            self.datacreator.boundary_variable = self.boundary_variable
            inp_data = self.datacreator.get_data()

            normals = self.setting.variables[self.boundary_variable] \
                .domain.boundary_normal(inp_data[self.boundary_variable])

            if self.data_fun is None:
                return {**inp_data,
                        'normal': normals}
            else:
                inp_data, data = apply_data_fun(self.data_fun,
                                                {**inp_data, 'normal': normals},
                                                whole_batch=self.data_fun_whole_batch,
                                                batch_size=self.dataset_len)
                return {**inp_data,
                        'data': data,
                        'normal': normals}
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['bound_condition_fun'] = self.bound_condition_fun.__name__
        if self.data_fun is not None:
            dct['data_fun'] = self.data_fun.__name__
        dct['dataset_size'] = self.datacreator.dataset_size
        dct['sampling_strategy'] = self.datacreator.sampling_strategy
        dct['boundary_sampling_strategy'] = self.datacreator.boundary_sampling_strategy
        return dct


def apply_data_fun(f, args, whole_batch=True, batch_size=None):
    # typical steps to apply a user-defined data function:
    # 1) filter the input arguments required by the given function
    inp = prepare_user_fun_input(f, args)
    # 2) if the function is defined entry-wise, we wrap it in a for loop
    if whole_batch:
        out = f(**inp)
    else:
        out = apply_to_batch(f, batch_size=batch_size, **inp)
    # 3) data points that evaluated to None or NaN should not be used
    if whole_batch:
        batch_size = len(out)
    args, out = remove_nan(args, out, batch_size)
    return args, out


def remove_nan(inp, out, batch_size):
    # remove input and output data where the operation evaluated to NaN
    keep = ~(np.isnan(out).any(axis=tuple(range(1, len(np.shape(out))))))
    if np.any(~keep):
        print(f"""Warning: {np.sum(~keep)} values will be removed from the data because
                  the given data_fun evaluated to None or NaN. Please make sure this is
                  the desired behaviour.""")
    for v in inp:
        if is_batch(inp[v], batch_size):
            inp[v] = inp[v][keep]
    out = out[keep]
    return inp, out


def get_data_len(size):
    if isinstance(size, int):
        return size
    elif isinstance(size, (tuple, list)):
        return np.prod(size)
    elif isinstance(size, dict):
        return np.prod(list(size.values()))
    else:
        raise ValueError(f"""'dataset_size should be one of int,
                             tuple, list or dict. Got {type(size)}.""")
