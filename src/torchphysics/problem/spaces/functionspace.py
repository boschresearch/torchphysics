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
