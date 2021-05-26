from .condition import Condition
from .variables import Variable


class Problem():
    """
    NOTE: the whole registering process for variables and conditions
    should be streamlined
    """
    def __init__(self, variables, train_conditions, val_conditions):
        # the problem, variables and conditions store a dict of all variables
        self.variables = self._create_dict(variables, Variable)
        # register those variables
        for name in self.variables:
            self.variables[name].context = self.variables

        # create dictionaries of conditions
        self.train_conditions = self._create_dict(train_conditions, Condition)
        for name in self.train_conditions:
            self.train_conditions[name].variables = self.variables

        self.val_conditions = self._create_dict(val_conditions, Condition)
        for name in self.val_conditions:
            self.train_conditions[name].variables = self.variables

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
        dct = self.train_conditions
        for vname in self.variables:
            vconditions = self.variables[vname].get_train_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def get_val_conditions(self):
        dct = self.val_conditions
        for vname in self.variables:
            vconditions = self.variables[vname].get_val_conditions()
            for cname in vconditions:
                name_str = f"{vname}_{cname}"
                assert name_str not in dct, \
                    f"{name_str} cannot be present twice."
                dct[name_str] = vconditions[cname]
        return dct

    def _create_dict(self, arg, _type):
        """create a dictionary of the elements in arg and check
        whether all elements are of the correct type"""

        if isinstance(arg, (list, tuple)):
            dct = {}
            for elem in arg:
                assert isinstance(elem, _type), f"{elem} is of the wrong type."
                assert elem.name not in dct, f"{elem.name} cannot be present twice."
                dct[elem.name] = elem
        elif isinstance(arg, dict):
            for key in arg:
                assert isinstance(arg[key], _type), f"{arg[key]} is of the wrong type."
            dct = arg
        else:
            raise TypeError(f"""Got type {type(arg)} but expected
                             one of list, tuple or dict.""")
        return dct

    def is_well_posed(self):
        raise NotImplementedError
