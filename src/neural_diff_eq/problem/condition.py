"""Conditions are the central concept in this package.
They supply the necessary training data to the model.
"""
import abc
import torch


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
        self.variables = None

    @abc.abstractmethod
    def get_data(self):
        """Creates and returns the data for the given condition."""
        return

    @abc.abstractmethod
    def get_data_plot_variables(self):
        return

    def is_registered(self):
        return self.variables is not None

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
    dataset_size : int
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
    """
    def __init__(self, pde, norm, name='pde',
                 sampling_strategy='random', weight=1.0, 
                 dataset_size=10000, track_gradients=True,
                 data_plot_variables=False):
        super().__init__(name, norm, weight,
                         track_gradients=track_gradients,
                         data_plot_variables=data_plot_variables)
        self.sampling_strategy = sampling_strategy
        self.pde = pde
        self.dataset_size = dataset_size

    def forward(self, model, data):
        u = model(data)
        err = self.pde(u, data)
        return self.norm(err, torch.zeros_like(err))

    def get_data(self):
        if self.is_registered():
            data = {}
            for vname in self.variables:
                data[vname] = self.variables[vname].domain.sample_inside(
                    self.dataset_size,
                    type=self.sampling_strategy
                )
            return data
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['sampling_strategy'] = self.sampling_strategy
        dct['pde'] = self.pde.__name__
        dct['dataset_size'] = self.dataset_size
        return dct

    def get_data_plot_variables(self):
        if self.data_plot_variables is True:
            return self.variables
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
        data, target = data
        u = model(data, track_gradients=False)
        return self.norm(u, target)

    def get_data(self):
        if self.is_registered():
            return (self.data_x, self.data_u)
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
    requires_input_grad : bool
        If True, the gradients are still tracked during validation to enable the
        computation of derivatives w.r.t. the inputs
    weight : float
        Scalar weight of this condition that is used in the weighted sum for the
        training loss. Defaults to 1.
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
        A method that takes boundary points (in the usual dictionary form) a an input
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
    dataset_size : int
        Amount of samples in the used dataset. The dataset is generated once at the
        beginning of the training.
    independent_of_model : bool
        Indicates if the condition can be computed without the output of the model.
        E.g. a inital condition for the model is independent of the output, a condition
        for the first derivative needs the output to compute the derivative.
    """
    def __init__(self, dirichlet_fun, name, norm,
                 sampling_strategy='random', boundary_sampling_strategy='random',
                 weight=1.0, dataset_size=10000,
                 data_plot_variables=True, independent_of_model=True):
        super().__init__(name, norm, weight=weight,
                         track_gradients=False, data_plot_variables=data_plot_variables)
        self.dirichlet_fun = dirichlet_fun
        self.boundary_sampling_strategy = boundary_sampling_strategy
        self.sampling_strategy = sampling_strategy
        self.dataset_size = dataset_size
        self.independent_of_model = independent_of_model

    def forward(self, model, data):
        data, target = data
        u = model(data)
        return self.norm(u, target)

    def get_data(self):
        if self.is_registered():
            data = {}
            for vname in self.variables:
                if vname == self.boundary_variable:
                    data[vname] = self.variables[vname].domain.sample_boundary(
                        self.dataset_size,
                        type=self.boundary_sampling_strategy
                    )
                else:
                    data[vname] = self.variables[vname].domain.sample_inside(
                        self.dataset_size,
                        type=self.sampling_strategy
                    )
            return (data, self.dirichlet_fun(data))
        else:
            raise RuntimeError("""Conditions need to be registered in a
                                  Variable or Problem.""")

    def serialize(self):
        dct = super().serialize()
        dct['dirichlet_fun'] = self.dirichlet_fun.__name__
        dct['dataset_size'] = self.dataset_size
        dct['sampling_strategy'] = self.sampling_strategy
        dct['boundary_sampling_strategy'] = self.boundary_sampling_strategy
        dct['dataset_size'] = self.dataset_size
        return dct
