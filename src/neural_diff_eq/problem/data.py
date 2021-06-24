import torch

print("""
      These classes are old and will be removed!
      """)

class Dataset(torch.utils.data.Dataset):
    """
    The standard dataset that samples data points in the inner or on the boundary
    of the domain.

    Parameters
    ----------
    variables : dict
        Dictionary of variable names and the Variable objects
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points for this
        condition. See domains for more details.
    boundary : str
        Name of the boundary variable, None if dataset should be sampled from inner of
        domain.
    size : int
        Amount of samples in the dataset.
    """
    def __init__(self, variables, sampling_strategy='random',
                 boundary_sampling_strategy=None, boundary=None, size=10000):
        super().__init__()
        self.variables = variables
        self.boundary = boundary
        self.sampling_strategy = sampling_strategy
        self.boundary_sampling_strategy = boundary_sampling_strategy
        self.size = size

        self.cache_dict = {}
        self._cache_items(self.size)

    def __len__(self):
        return self.size

    def _cache_items(self, n):
        for vname in self.variables:
            if vname == self.boundary:
                self.cache_dict[vname] = self.variables[vname].domain.sample_boundary(
                    n, type=self.boundary_sampling_strategy)
            else:
                self.cache_dict[vname] = self.variables[vname].domain.sample_inside(
                    n, type=self.sampling_strategy)

    def __getitem__(self, index):
        dct = {}
        for vname in self.cache_dict:
            dct[vname] = self.cache_dict[vname][index]
        return dct


class DataDataset(torch.utils.data.Dataset):
    """
    The standard dataset that samples data points in the inner or on the boundary
    of the domain.

    Parameters
    ----------
    variables : dict
        Dictionary of variable names and the Variable objects
    data_x : dict
        A dictionary containing pairs of variables and data for that variables,
        organized in numpy arrays or torch tensors of equal length.
    data_u : array-like
        The targeted solution values for the data points in data_x.
    """
    def __init__(self, variables, data_x, data_u):
        super().__init__()
        self.variables = variables
        self.data_x = data_x
        self.data_u = data_u
        self.size = len(data_u[:, 0])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        dct = {}
        for vname in self.data_x:
            dct[vname] = self.data_x[vname][index]
        return dct, self.data_u[index]


class FunctiondataDataset(torch.utils.data.Dataset):
    """
    A dataset that samples data points in the inner or on the boundary
    of the domain and computes/saves the desired function values a those points.

    Parameters
    ----------
    variables : dict
        Dictionary of variable names and the Variable objects
    function : function handle
        A method that takes boundary points (in the usual dictionary form) as an input
        and returns the desired function values at those points.
    sampling_strategy : str
        The sampling strategy used to sample data points for this condition. See domains
        for more details.
    boundary_sampling_strategy : str
        The sampling strategy used to sample the boundary variable's points for this
        condition. See domains for more details.
    boundary : str
        Name of the boundary variable, None if dataset should be sampled from inner of
        domain.
    size : int
        Amount of samples in the dataset.
    """
    def __init__(self, variables, function, sampling_strategy='random',
                 boundary_sampling_strategy=None, boundary=None, size=10000):
        super().__init__()
        self.variables = variables
        self.boundary = boundary
        self.sampling_strategy = sampling_strategy
        self.boundary_sampling_strategy = boundary_sampling_strategy
        self.size = size

        self.cache_dict = {}
        self._cache_items(self.size)
        self.function_data = function(self.cache_dict)

    def __len__(self):
        return self.size

    def _cache_items(self, n):
        for vname in self.variables:
            if vname == self.boundary:
                self.cache_dict[vname] = torch.from_numpy(
                    self.variables[vname].domain.sample_boundary(
                        n, type=self.boundary_sampling_strategy))
            else:
                self.cache_dict[vname] = torch.from_numpy(
                    self.variables[vname].domain.sample_inside(
                        n, type=self.sampling_strategy))

    def __getitem__(self, index):
        dct = {}
        for vname in self.cache_dict:
            dct[vname] = self.cache_dict[vname][index]
        return dct, self.function_data[index]