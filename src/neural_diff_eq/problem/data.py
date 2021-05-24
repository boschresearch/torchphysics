import torch


class DiffEqDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError
