
class FunctionSpace():
    """
    A FunctionSpace collects functions that map from a specific input domain
    to a previously defined output space.

    Parameters
    ----------
    input_domain : torchphysics.Domain
        The input domain of the functions in this function space.
    output_space : torchphysics.Space
        The space of the image of the functions in this function space.
    """
    def __init__(self, input_domain, output_space):
        self.input_domain = input_domain
        self.output_space = output_space
        self.input_space = self.input_domain.space
