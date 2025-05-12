class FunctionSpace:
    """
    A FunctionSpace collects functions that map from a specific input domain
    to a previously defined output space.

    Parameters
    ----------
    input_space : torchphysics.Space
        The space of the input domain of the functions in this function space.
    output_space : torchphysics.Space
        The space of the image of the functions in this function space.
    """

    def __init__(self, input_space, output_space):
        self.input_space = input_space
        self.output_space = output_space


    def __mul__(self, other):
        """Creates the product space of the two input spaces. Allows the
        construction of higher dimensional spaces with 'mixed' variable names.
        E.g R1('x')*R1('y') is a two dimensional space where one axis is 'x'
        and the other stands for 'y'.
        """
        assert isinstance(other, FunctionSpace), "Can only multiply FunctionSpaces"
        assert not (self.output_space == other.output_space), "Output spaces must be different"
        if self.input_space == other.input_space:
            return FunctionSpace(self.input_space, self.output_space * other.output_space)
        elif self.input_space in other.input_space:
            return FunctionSpace(other.input_space, self.output_space * other.output_space)
        elif other.input_space in self.input_space:
            return FunctionSpace(self.input_space, self.output_space * other.output_space)
        else:
            return FunctionSpace(self.input_space*other.input_space, 
                                 self.output_space * other.output_space)
