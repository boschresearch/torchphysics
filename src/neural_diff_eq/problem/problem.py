from .condition import DiffEqCondition
from .variables import Variable


class DiffEqProblem():
    def __init__(self, variables, train_conditions, val_conditions=None):
        # create a dictionary of variables
        self.variables = self._create_dict(variables, Variable)

        # create a dictionary of conditions
        self.train_conditions = self._create_dict(train_conditions, DiffEqCondition)
        # val_conditions could also be left empty
        if val_conditions is not None:
            self.val_conditions = self._create_dict(val_conditions, DiffEqCondition)
        else:
            self.val_conditions = val_conditions

    def add_train_condition(self, condition, var=None):
        if var is None:
            self.train_conditions.append(condition)
        else:
            self.variables[var].add_train_condition(condition)

    def add_val_condition(self, condition, var=None):
        if var is None:
            if self.val_conditions is None:
                self.val_conditions = [condition]
            else:
                self.val_conditions.append(condition)
        else:
            self.variables[var].add_val_condition(condition)

    def get_train_conditions(self):
        dct = self.train_conditions
        for vname in self.variables:
            vconditions = self.variables[vname].get_train_conditions()
            for cname in vconditions:
                dct[f"{vname}_{cname}"] = vconditions[cname]
        return dct

    def get_val_conditions(self):
        if self.val_conditions is None:
            dct = self.train_conditions
        else:
            dct = self.val_conditions
        for vname in self.variables:
            vconditions = self.variables[vname].get_val_conditions()
            for cname in vconditions:
                dct[f"{vname}_{cname}"] = vconditions[cname]
        return dct
    
    def _create_dict(self, arg, _type):
        """create a dictionary of the elements in arg and check
        whether all elements are of the correct type"""

        if isinstance(arg, (list, tuple)):
            dct = {}
            for elem in arg:
                assert isinstance(elem, _type), f"{elem} is of the wrong type."
                dct[elem.name] = elem
        elif isinstance(arg, dict):
            for key in arg:
                assert isinstance(arg[key], _type), f"{arg[key]} is of the wrong type."
            dct = arg
        else:
            raise TypeError(f"""Got type {type(arg)} but expected
                             one of list, tuple or dict.""")
        return dct
