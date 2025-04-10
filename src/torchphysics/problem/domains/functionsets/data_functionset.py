import torch

from .functionset import DiscreteFunctionSet
from ...spaces import Points


class DataFunctionSet(DiscreteFunctionSet):
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
        assert data.shape[-1] == function_space.output_space.dim
        super().__init__(function_space, len(data), data.shape[1:-1])
        self.data = data
    
    def create_functions(self, device="cpu"):
        self.data = self.data.to(device)

    def get_function(self, idx):
        return Points(self.data[idx], self.function_space.output_space)
    
    def compute_normalization(self):
        self.mean_tensor = torch.mean(self.data, dim=0, keepdim=True)
        self.std_tensor = torch.std(self.data, dim=0, keepdim=True)

    def compute_pca(self, components, normalize_data = True):
        data_copy = self.data
        if normalize_data:
            data_copy = (self.data - self.mean) / self.std

        self.pca = torch.pca_lowrank(torch.flatten(data_copy, 1), 
                                     q=components)