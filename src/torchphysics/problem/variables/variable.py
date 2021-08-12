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
    order : int
        The order of the problem w.r.t this variable.
    """
    def __init__(self, name, domain, train_conditions={}, val_conditions={}, order=0):
        self.setting = None  # other variables, are set by Setting object
        self.name = name
        self.domain = domain
        self.order = order
        super().__init__(train_conditions=train_conditions,
                         val_conditions=val_conditions)

    def add_train_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.train_conditions
        condition.setting = self.setting
        condition.boundary_variable = self.name
        self.train_conditions[condition.name] = condition

    def add_val_condition(self, condition):
        assert isinstance(condition, BoundaryCondition), """Variables can only
            handle boundary conditions."""
        assert condition.name not in self.val_conditions
        condition.setting = self.setting
        condition.boundary_variable = self.name
        self.val_conditions[condition.name] = condition

    def get_train_conditions(self):
        return self.train_conditions

    def get_val_conditions(self):
        return self.val_conditions

    def is_well_posed(self):
        raise NotImplementedError

    def get_dim(self):
        return self.domain.dim

    def serialize(self):
        dct = {}
        dct['name'] = self.name
        dct['domain'] = self.domain.serialize()
        t_c_dict = {}
        for c_name in self.train_conditions:
            t_c_dict[c_name] = self.train_conditions[c_name].serialize()
        dct['train_conditions'] = t_c_dict
        v_c_dict = {}
        for c_name in self.val_conditions:
            v_c_dict[c_name] = self.val_conditions[c_name].serialize()
        dct['val_conditions'] = v_c_dict
        return dct
