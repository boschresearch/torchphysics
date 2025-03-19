
from .functionset import FunctionSet
from ...spaces import Points


class DataFunctionSet(FunctionSet):
    """FunctionSet that is created from a given data set.
    This function set is always a discret set, since the data can not
    be evaluated at arbitrary points.
    
    Parameters
    ----------
    function_space : tp.spaces.FunctionSpace
        The function space of the functions in the set.
    data : torch.Tensor
        The data that describes the function values. The shape of the last 
        dimension has to match the dimension of the output space. 
    """
    def __init__(self, function_space, data):
        super().__init__(function_space, len(data))
        assert data.shape[-1] == self.function_space.output_space.dim
        self.data = data
    
    @property
    def is_discretized(self):
        return True
    
    def create_functions(self, device="cpu"):
        self.data = self.data.to(device)

    def get_function(self, idx):
        return Points(self.data[idx], self.function_space.output_space)