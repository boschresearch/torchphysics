import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, variables, sampling_strategy='random',
                 boundary=None, size=10000):
        super().__init__()
        self.variables = variables
        self.boundary = boundary
        self.sampling_strategy = sampling_strategy
        self.size = size

        self.cache_dict = {}
        self._cache_items(self.size)

    def __len__(self):
        return self.size

    def _cache_items(self, n):
        for vname in self.variables:
            if vname == self.boundary:
                self.cache_dict[vname] = self.variables[vname].domain.sample_boundary(
                    n, type=self.sampling_strategy)
            else:
                self.cache_dict[vname] = self.variables[vname].domain.sample_inside(
                    n, type=self.sampling_strategy)

    def __getitem__(self, index):
        dct = {}
        for vname in self.cache_dict:
            dct[vname] = self.cache_dict[vname][index]


class DataDataset(torch.utils.data.Dataset):
    def __init__(self, variables, data_x, data_u):
        super().__init__()
