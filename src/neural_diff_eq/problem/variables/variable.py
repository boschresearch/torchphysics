from ..problem import Problem
from ..condition import BoundaryCondition


class Variable(Problem):

    def __init__(self, name, domain, train_conditions, val_conditions, order=0):
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)
        self.name = name
        self.domain = domain
        self.order = order
        self.context = None  # other variables, are set by problem object

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
    NOTE: the whole registering process for variables and conditions
    should be streamlined
    """
    def __init__(self, variables, train_conditions, val_conditions):
        # the problem, variables and conditions store a dict of all variables
        self.variables = self._create_dict(variables, Variable)
        # register those variables
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)
        for vname in self.variables:
            self.variables[vname].context = self.variables
            for cname in self.variables[vname].get_train_conditions():
                self.variables[vname].get_train_conditions()[cname].variables = self.variables
            for cname in self.variables[vname].get_val_conditions():
                self.variables[vname].get_val_conditions()[cname].variables = self.variables

    def add_train_condition(self, condition, boundary_var=None):
        if boundary_var is None:
            assert condition.name not in self.train_conditions, \
                f"{condition.name} cannot be present twice."
            condition.variables = self.variables
            self.train_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_train_condition(condition)

    def add_val_condition(self, condition, boundary_var=None):
        if boundary_var is None:
            assert condition.name not in self.val_conditions, \
                f"{condition.name} cannot be present twice."
            condition.variables = self.variables
            self.val_conditions[condition.name] = condition
        else:
            self.variables[boundary_var].add_val_condition(condition)

    def get_train_conditions(self):
        """Returns all training conditions present in this problem.
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
