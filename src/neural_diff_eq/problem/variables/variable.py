from ..condition import BoundaryCondition


class Variable():

    def __init__(self, name, domain, order=0, conditions={}):

        self.name = name
        self.domain = domain
        self.order = order
        self.context = None  # other variables, are set by problem object
        self.conditions = conditions

    def add_training_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.conditions
        condition.variables = self.context
        condition.boundary_variable = self.name
        self.conditions[condition.name] = condition
