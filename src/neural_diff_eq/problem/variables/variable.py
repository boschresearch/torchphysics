from ..problem import Problem
from ..condition import BoundaryCondition


class Variable(Problem):
    """
    The problem associated to a single variable.

    Parameters
    ----------
    name : str
        Name of this variable. By this name, the variable will be found in settings
        or conditions.
    domain : Domain
        A Domain object, describing the used domain of this variable
    train_conditions : list or dict of conditions
        Boundary conditions for this variable that are used in training
    val_conditions : list or dict of conditions
        Boundary conditions for this variable that are tracked during validation
    """
    def __init__(self, name, domain, train_conditions={}, val_conditions={}, order=0):
        self.context = None  # other variables, are set by Setting object
        self.name = name
        self.domain = domain
        self.order = order
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)

    def add_train_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.train_conditions
        condition.variables = self.context
        condition.boundary_variable = self.name
        self.train_conditions[condition.name] = condition

    def add_val_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.val_conditions
        condition.variables = self.context
        condition.boundary_variable = self.name
        self.val_conditions[condition.name] = condition

    def get_train_conditions(self):
        return self.train_conditions

    def get_val_conditions(self):
        return self.val_conditions

    def is_well_posed(self):
        raise NotImplementedError


class Setting(Problem):
    """
    A PDE setting, built of all variables and conditions in this problem.

    Parameters
    ----------
    variables : dict, tuple, list or Variable
        A collection of Variables for this DE Problem. The Domain of the Problem is
        the cartesian product of the domains of the variables.
    train_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are used in training
    val_conditions : list or dict of conditions
        Conditions on the inner part of the domain that are tracked during validation
    """
    def __init__(self, variables={}, train_conditions={}, val_conditions={}):
        self.variables = {}
        if isinstance(variables, (list, tuple)):
            for condition in variables:
                self.add_variable(condition)
        elif isinstance(variables, dict):
            for condition_name in variables:
                self.add_variable(variables[condition_name])
        elif isinstance(variables, Variable):
            self.add_variable(variables)
        else:
            raise TypeError(f"""Got type {type(variables)} but expected
                             one of list, tuple, dict or Variable.""")
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)

    def add_variable(self, variable):
        """Adds a new Variable object to the Setting, registers the variable and its conditions
        """
        assert isinstance(variable, Variable), f"{variable} should be a Variable obj."
        self.variables[variable.name] = variable
        # register the variable in this setting
        variable.context = self.variables
        # update its condition variables
        for cname in variable.train_conditions:
            variable.train_conditions[cname].variables = self.variables
        for cname in variable.val_conditions:
            variable.val_conditions[cname].variables = self.variables

    def add_train_condition(self, condition, boundary_var=None):
        """Adds and registers a condition that is used during training
        """
        if boundary_var is None:
            assert condition.name not in self.train_conditions, \
                f"{condition.name} cannot be present twice."
            condition.variables = self.variables
            self.train_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_train_condition(condition)

    def add_val_condition(self, condition, boundary_var=None):
        """Adds and registers a condition that is used for validation
        """
        if boundary_var is None:
            assert condition.name not in self.val_conditions, \
                f"{condition.name} cannot be present twice."
            condition.variables = self.variables
            self.val_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_val_condition(condition)

    def get_train_conditions(self):
        """Returns all training conditions present in this problem.
        This does also include conditions in the variables.
        """
        dct = {}
        for cname in self.train_conditions:
            dct[cname] = self.train_conditions[cname]
        for vname in self.variables:
            vconditions = self.variables[vname].get_train_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def get_val_conditions(self):
        """Returns all validation conditions present in this problem.
        This does also include conditions in the variables.
        """
        dct = {}
        for cname in self.val_conditions:
            dct[cname] = self.val_conditions[cname]
        for vname in self.variables:
            vconditions = self.variables[vname].get_val_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def is_well_posed(self):
        raise NotImplementedError
