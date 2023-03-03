import math
import torch

from ...problem.spaces import Points

class PointsDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to load tuples of data points.

    Parameters
    ----------
    data_points : Points or tuple
        One or multiple Points object containing multiple data points or tuples of data points.
        If a tuple of Points objects is given, they should all have the same length, as data will
        be loaded in tuples where the i-th points are loaded simultaneously.
    batch_size : int
        The size of the loaded batches.
    shuffle : bool
        Whether to shuffle the order of the data points at initialization.
    drop_last : bool
        Whether to drop the last (and non-batch-size-) minibatch.
    """
    def __init__(self, data_points, batch_size, shuffle=False, drop_last=False):
        if isinstance(data_points, Points):
            self.data_points = [data_points]
        else:
            assert isinstance(data_points, (tuple, list))
            self.data_points = list(data_points)
        if shuffle:
            perm = torch.randperm(len(self.data_points[0]))
            for i in range(len(self.data_points)):
                self.data_points[i] = self.data_points[i][perm]

        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __len__(self):
        """Returns the number of points of this dataset.
        """
        if self.drop_last:
            return len(self.data_points[0]) // self.batch_size
        else:
            return math.ceil(len(self.data_points[0]) / self.batch_size)

    def __getitem__(self, idx):
        """Returns the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the desired point.
        """
        l = len(self.data_points[0])
        out = []
        for points in self.data_points:
            out.append(points[idx*self.batch_size:min((idx+1)*self.batch_size, l), :])
        return tuple(out)

class PointsDataLoader(torch.utils.data.DataLoader):
    """
    A DataLoader that can be used in a condition to load minibatches of paired data
    points as the input and output of a model.

    Parameters
    ----------
    data_points : Points or tuple
        One or multiple Points object containing multiple data points or tuples of data points.
        If a tuple of Points objects is given, they should all have the same length, as data will
        be loaded in tuples where the i-th points are loaded simultaneously.
    batch_size : int
        The size of the loaded batches.
    shuffle : bool
        Whether to shuffle the order of the data points at initialization.
    num_workers : int
        The amount of workers used during data loading, see also: the PyTorch documentation
    pin_memory : bool
        Whether to use pinned memory during data loading, see also: the PyTorch documentation
    drop_last : bool
        Whether to drop the last (and non-batch-size-) minibatch.
    """
    def __init__(self, data_points, batch_size, shuffle=False,
                 num_workers=0, pin_memory=False, drop_last=False):
        super().__init__(PointsDataset(data_points, batch_size,
                                       shuffle=shuffle, drop_last=drop_last),
                         batch_size=None,
                         shuffle=False,
                         num_workers=num_workers,
                         pin_memory=pin_memory)

