
class FunctionSet():
    """
    A set of functions that can be sampled to supply samples from a function space.
    """
    def __init__(self, function_space, parameter_domain):
        self.function_space = function_space
        self.parameter_domain = parameter_domain
    
    def __add__(self, other):
        """
        Combines two function sets.
        """