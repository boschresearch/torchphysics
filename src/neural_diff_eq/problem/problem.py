import abc

from .condition import Condition


class Problem():
    def __init__(self, train_conditions, val_conditions):
        # create dictionaries of conditions
        self.train_conditions = self._create_dict(train_conditions, Condition)
        for name in self.train_conditions:
            self.train_conditions[name].variables = self.variables

        self.val_conditions = self._create_dict(val_conditions, Condition)
        for name in self.val_conditions:
            self.val_conditions[name].variables = self.variables

    @abc.abstractmethod
    def add_train_condition(self, condition):
        pass

    @abc.abstractmethod
    def add_val_condition(self, condition):
        pass

    @abc.abstractmethod
    def get_train_conditions(self):
        pass

    @abc.abstractmethod
    def get_val_conditions(self):
        pass

    @abc.abstractmethod
    def is_well_posed(self):
        pass

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
