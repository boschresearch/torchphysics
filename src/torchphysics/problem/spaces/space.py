import torch
from collections import Counter, OrderedDict


class Space(Counter, OrderedDict):
    """A Space defines (and assigns) the dimensions of the variables
    that appear in the differentialequation. This class sholud not be instanced
    directly, rather the corresponding child classes.

    Parameters
    ----------
    variables_dims : dict
        A dictionary containing the name of the variables and the dimension
        of the respective variable.
    """

    def __init__(self, variables_dims):
        # set counter of variable names and their dimensionalities
        super().__init__(variables_dims)

    def cast_tensor_into_space(self, data_tensor):
        """Applies transformations to the tensor such that the constrains 
        of the space are fulfilled. E.g. rounds the input to integer values.

        Parameters
        ----------
        data_tensor : torch.tensor
            The tensor which data should be casted, such that it fulfills the
            constrains of this space.
        """
        return data_tensor

    def __mul__(self, other):
        """Creates the product space of the two input spaces. Allows the
        construction of higher dimensional spaces with 'mixed' variable names.
        E.g R1('x')*R1('y') is a two dimensional space where one axis is 'x'
        and the other stands for 'y'.
        """
        assert isinstance(other, Space)
        return Space(self + other)

    def __contains__(self, space):
        """Checks if the variables of the other space are contained in this
        space.

        Parameters
        ----------
        space : torchphysics.spaces.Space
            The other Space that should be checked if this is included.
        """
        if isinstance(space, str):
            return super().__contains__(space)
        if isinstance(space, Space):
            return (self & space) == space
        else:
            return False

    def __getitem__(self, val):
        """Returns a part of the Space dicitionary, specified in the
        input. Mathematically, this constructs a subspace.

        Parameters
        ----------
        val : str, slice, list or tuple
            The keys that correspond to the variables that should be used in the
            subspace.
        """
        if isinstance(val, slice):
            keys = list(self.keys())
            new_slice = slice(
                keys.index(val.start) if val.start is not None else None,
                keys.index(val.stop) if val.stop is not None else None,
                val.step,
            )
            new_keys = keys[new_slice]
            return Space({k: self[k] for k in new_keys})
        if isinstance(val, list) or isinstance(val, tuple):
            return Space({k: self[k] for k in val})
        else:
            return super().__getitem__(val)

    @property
    def dim(self):
        """Returns the dimension of the space (sum of factor spaces)"""
        return sum(self.values())

    @property
    def variables(self):
        """
        A unordered (!) set of variables.
        """
        return set(self.keys())

    def __eq__(self, o: object) -> bool:
        # use OrderedDict equal methode to get order-sensitive comparision
        return OrderedDict.__eq__(self, o)

    def __ne__(self, o: object) -> bool:
        return OrderedDict.__ne__(self, o)

    """
    Python recipe (see official Python docs) to maintain the insertion order.
    This way, dimensions with identical variable names will be joined, all
    other dimensions will be kept in the order of their creation by products
    or __init__.
    """

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, dict(OrderedDict(self)))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

    def check_values_in_space(self, values):
        """Checks if a given tensor is valid to belong to this space.

        Parameters
        ----------
        values : torch.tensor
            A tensor of values that should be checked.
            Generally the last dimension of the tensor has to fit
            the dimension of this space.

        Returns
        -------
        torch.tensor
            In the case, that the values have not the corrected shape, but can
            be reshaped, thet reshaped values are returned.
            This is used in the matrix-space.
        """
        assert values.shape[-1] == self.dim
        return values


class R1(Space):
    """The space for one dimensional real numbers.

    Parameters
    ----------
    variable_name: str
        The name of the variable that belongs to this space.
    """

    def __init__(self, variable_name):
        super().__init__({variable_name: 1})


class R2(Space):
    """The space for two dimensional real numbers.

    Parameters
    ----------
    variable_name: str
        The name of the variable that belongs to this space.
    """

    def __init__(self, variable_name):
        super().__init__({variable_name: 2})


class R3(Space):
    """The space for three dimensional real numbers.

    Parameters
    ----------
    variable_name: str
        The name of the variable that belongs to this space.
    """

    def __init__(self, variable_name):
        super().__init__({variable_name: 3})


class Rn(Space):
    """The space for n dimensional real numbers.

    Parameters
    ----------
    variable_name: str
        The name of the variable that belongs to this space.
    n : int
        The dimension of this space.
    """

    def __init__(self, variable_name, n: int):
        super().__init__({variable_name: n})


class IntegerSpace(Space):

    def __init__(self, variable_name, dim):
        super().__init__({variable_name: dim})

    def cast_tensor_into_space(self, data_tensor):
        return torch.round(data_tensor) # to keep same type


class Z1(IntegerSpace):
    """The space for one dimensional integer numbers.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 1)

class Z2(IntegerSpace):
    """The space for two dimensional integer numbers.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 2)

class Z3(IntegerSpace):
    """The space for three dimensional integer numbers.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 3)

class Zn(IntegerSpace):
    """The space for n dimensional integer numbers.
    """
    def __init__(self, variable_name, n):
        super().__init__(variable_name, n)


class NaturalNumberSpace(Space):
    def __init__(self, variable_name, dim):
        super().__init__({variable_name: dim})

    def cast_tensor_into_space(self, data_tensor):
        return torch.round(data_tensor).abs() # to keep same type

class N1(NaturalNumberSpace):
    """The space for one dimensional natural numbers including 0.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 1)

class N2(NaturalNumberSpace):
    """The space for two dimensional natural numbers including 0.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 2)

class N3(NaturalNumberSpace):
    """The space for three dimensional natural numbers including 0.
    """
    def __init__(self, variable_name):
        super().__init__(variable_name, 3)

class Nn(NaturalNumberSpace):
    """The space for n dimensional natural numbers including 0.
    """
    def __init__(self, variable_name, n):
        super().__init__(variable_name, n)