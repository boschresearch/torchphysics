import math
import torch
import numpy as np

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


class DeepONetDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to load tuples of data points, used via DeepONetDataLoader
    """
    def __init__(self, branch_data_points, trunk_data_points, out_data_points,
                 branch_space, trunk_space, output_space,
                 branch_batch_size, trunk_batch_size, shuffle_branch=False,
                 shuffle_trunk=True):

        assert out_data_points.shape[0] == branch_data_points.shape[0]
        assert out_data_points.shape[1] == trunk_data_points.shape[0]

        self.trunk_data_points = trunk_data_points
        self.branch_data_points = branch_data_points
        self.out_data_points = out_data_points

        if shuffle_trunk:
            trunk_perm = torch.randperm(len(self.trunk_data_points))
            self.trunk_data_points = self.trunk_data_points[trunk_perm]
            self.out_data_points = self.out_data_points[:, trunk_perm]
        if shuffle_branch:
            branch_perm = torch.randperm(len(self.branch_data_points))
            self.branch_data_points = self.branch_data_points[branch_perm]
            self.out_data_points = self.out_data_points[branch_perm, :]

        self.trunk_batch_size = len(self.trunk_data_points) if trunk_batch_size < 0 else trunk_batch_size
        self.branch_batch_size = len(self.branch_data_points) if branch_batch_size < 0 else branch_batch_size

        self.branch_space = branch_space
        self.trunk_space = trunk_space
        self.output_space = output_space
    
    def __len__(self):
        """Returns the number of points of this dataset.
        """
        # the least common multiple of both possible length will lead to the correct distribution
        # of data points and hopefully managable effort
        return int(np.lcm(
            int(np.lcm(len(self.branch_data_points), self.branch_batch_size) / self.branch_batch_size),
            int(np.lcm(len(self.trunk_data_points), self.trunk_batch_size) / self.trunk_batch_size)))

    def _slice_points(self, points, out_points, out_axis, batch_size, idx):
        a = (idx*batch_size) % len(points)
        b = ((idx+1)*batch_size) % len(points)
        if a < b:
            points = points[a:b]
            if out_axis == 0:
                out_points = out_points[a:b, :]
            elif out_axis == 1:
                out_points = out_points[:, a:b]
            else:
                raise ValueError
        else:
            points = torch.cat([points[a:], points[:b]], dim=0)
            if out_axis == 0:
                out_points = torch.cat([out_points[a:,:], out_points[:b,:]], dim=0)
            elif out_axis == 1:
                out_points = torch.cat([out_points[:,a:], out_points[:,:b]], dim=1)
            else:
                raise ValueError
        return points, out_points

    def __getitem__(self, idx):
        """Returns the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the desired point.
        """
        branch_points, out_points = self._slice_points(self.branch_data_points,
                                                       self.out_data_points,
                                                       0,
                                                       self.branch_batch_size,
                                                       idx)
        trunk_points, out_points = self._slice_points(self.trunk_data_points,
                                                      out_points,
                                                      1,
                                                      self.trunk_batch_size,
                                                      idx)
        return (Points(branch_points, self.branch_space),
                Points(trunk_points, self.trunk_space),
                Points(out_points, self.output_space))


class DeepONetDataLoader(torch.utils.data.DataLoader):
    """
    A DataLoader that can be used in a condition to load minibatches of paired data
    points as the input and output of a DeepONet-model.

    Parameters
    ----------
    branch_data : torch.tensor
        A tensor containing the input data for the branch network. Shape of the 
        data should be: [number_of_functions, input_dim_of_branch_net]
    trunk_data : torch.tensor
        A tensor containing the input data for the trunk network. Shape of the 
        data should be: 
        [number_of_discrete_points, input_dim_of_trunk_net]
        For each input of the branch_data we will have multiple inputs for the 
        trunk net.
    output_data : torch.tensor
        A tensor containing the expected output of the network. Shape of the 
        data should be: 
        [number_of_functions, number_of_discrete_points, output_dim].
    input_space : torchphysics.spaces.Space
        The input space of the trunk network.
    output_space : torchphysics.spaces.Space
        The output space in which the solution is. 
    batch_size : int
        The size of the loaded batches.
    shuffle_branch : bool
        Whether to shuffle the order of the branch functions at initialization.
    shuffle_trunk : bool
        Whether to shuffle the order of the trunk points at initialization.
    num_workers : int
        The amount of workers used during data loading, see also: the PyTorch documentation
    pin_memory : bool
        Whether to use pinned memory during data loading, see also: the PyTorch documentation
    drop_last : bool
        Whether to drop the last (and non-batch-size-) minibatch.
    """
    def __init__(self, branch_data, trunk_data, output_data, branch_space,
                 trunk_space, output_space, branch_batch_size, trunk_batch_size,
                 shuffle_branch=False, shuffle_trunk=True, num_workers=0,
                 pin_memory=False):
        trunk_data = trunk_data
        branch_data = branch_data
        output_data = output_data
        super().__init__(DeepONetDataset(branch_data,
                                         trunk_data,
                                         output_data,
                                         branch_space,
                                         trunk_space,
                                         output_space,
                                         branch_batch_size,
                                         trunk_batch_size,
                                         shuffle_branch,
                                         shuffle_trunk),
                         batch_size=None,
                         shuffle=False,
                         num_workers=num_workers,
                         pin_memory=pin_memory)