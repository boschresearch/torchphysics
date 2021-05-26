from ..problem import Problem
from ..condition import BoundaryCondition


class Variable(Problem):

    def __init__(self, name, domain, order=0, train_conditions={}, val_conditions={}):
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)
        self.name = name
        self.domain = domain
        self.order = order
        self.context = None  # other variables, are set by problem object

    def add_train_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.conditions
        condition.variables = self.context
        condition.boundary_variable = self.name
        self.train_conditions[condition.name] = condition

    def add_val_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.conditions
        condition.variables = self.context
        condition.boundary_variable = self.name
        self.val_conditions[condition.name] = condition

    def get_train_conditions(self):
        return self.train_conditions

    def get_val_conditions(self):
        return self.val_conditions

    def is_well_posed(self):
        raise NotImplementedError
