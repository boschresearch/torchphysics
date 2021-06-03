import abc

from .condition import Condition


class Problem():
    """Parent class for DE problems that have training and validation conditions.
    Child classes could be variables or whole PDE settings.

    Parameters
    ----------
    train_conditions : list or dict of conditions
        Conditions that will be used in the training process of this problem
    val_conditions : list or dict of conditions
        Conditions that will be used in the validation process of this problem
    """

    def __init__(self, train_conditions, val_conditions):
        # create dictionaries of conditions
        self.train_conditions = {}
        self._add_conditions(train_conditions, self.add_train_condition)
        self.val_conditions = {}
        self._add_conditions(val_conditions, self.add_val_condition)

    def _add_conditions(self, conditions, add_c_handle):
        """
        Helper function to add multiple conditions to the problem by calling
        add_x_condition() of the subclass.
        """
        if isinstance(conditions, (list, tuple)):
            for condition in conditions:
                add_c_handle(condition)
        elif isinstance(conditions, dict):
            for condition_name in conditions:
                add_c_handle(conditions[condition_name])
        elif isinstance(conditions, Condition):
            add_c_handle(conditions)
        else:
            raise TypeError(f"""Got type {type(conditions)} but expected
                             one of list, tuple, dict or Condition.""")

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
