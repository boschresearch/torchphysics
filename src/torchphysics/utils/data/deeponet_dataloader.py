import torch
import numpy as np

from ...problem.spaces import Points



class DeepONetDataLoader(torch.utils.data.DataLoader):
    """
    A DataLoader that can be used in a condition to load minibatches of paired data
    points as the input and output of a DeepONet-model.

    Parameters
    ----------
    branch_data : torch.tensor
        A tensor containing the input data for the branch network. Has to be of the shape:
        [number_of_functions, discrete_points_of_branch_net, function_space_dim]
        For example, if we have a batch of 20 vector-functions (:math:`f:\R \to \R^2`) and
        use 100 discrete points for the evaluation (where the branch nets evaluates f), 
        the shape would be: [20, 100, 2] 
    trunk_data : torch.tensor
        A tensor containing the input data for the trunk network. There are two different
        possibilites for the shape of this data:
            1) Every branch input function uses the same trunk values, then we can pass in
               the shape: [number_of_trunk_points, input_dim_of_trunk_net]
               This can speed up the trainings process.
            2) Or every branch function has different values for the trunk net, then we 
               need the shape: 
               [number_of_functions, number_of_trunk_points, input_dim_of_trunk_net]
               If this is the case, remember to set 'trunk_input_copied = false' inside
               the trunk net, to get the right trainings process.
    output_data : torch.tensor
        A tensor containing the expected output of the network. Shape of the 
        data should be: 
        [number_of_functions, number_of_trunk_points, output_dim].
    branch_output_space : torchphysics.spaces.Space
        The output space of the functions, that are used as the branch input.
    input_space : torchphysics.spaces.Space
        The input space of the trunk network.
    output_space : torchphysics.spaces.Space
        The output space in which the solution is. 
    branch_batch_size, trunk_batch_size : int
        The size of the loaded batches for trunk and branch.
    shuffle_branch : bool
        Whether to shuffle the order of the branch functions at initialization.
    shuffle_trunk : bool
        Whether to shuffle the order of the trunk points at initialization.
    num_workers : int
        The amount of workers used during data loading, see also: the PyTorch documentation
    pin_memory : bool
        Whether to use pinned memory during data loading, see also: the PyTorch documentation
    """
    def __init__(self, branch_data, trunk_data, output_data, branch_space,
                 trunk_space, output_space, branch_batch_size, trunk_batch_size,
                 shuffle_branch=False, shuffle_trunk=True, num_workers=0,
                 pin_memory=False):
        if len(trunk_data.shape) == 3:
            super().__init__(DeepONetDataset_Unique(branch_data,
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
        else:
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


class DeepONetDataset_Unique(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to load tuples of data points, used in the DeepONetDataLoader.
    Is used when every branch input has unique trunk inputs 
    -> Ordering of points is important.
    """
    def __init__(self, branch_data_points, trunk_data_points, out_data_points,
                 branch_space, trunk_space, output_space,
                 branch_batch_size, trunk_batch_size, shuffle_branch=False,
                 shuffle_trunk=True):

        assert out_data_points.shape[0] == branch_data_points.shape[0], \
            "Output values and branch inputs don't match!"
        assert trunk_data_points.shape[0] == branch_data_points.shape[0], \
            "Trunk and branch batch does not match!"
        assert out_data_points.shape[1] == trunk_data_points.shape[1], \
            "Output values and trunk inputs don't match!"

        self.trunk_data_points = trunk_data_points
        self.branch_data_points = branch_data_points
        self.out_data_points = out_data_points

        if shuffle_trunk:
            trunk_perm = torch.randperm(len(self.trunk_data_points[0]))
            self.trunk_data_points = self.trunk_data_points[:, trunk_perm, :]
            self.out_data_points = self.out_data_points[:, trunk_perm, :]
        if shuffle_branch:
            branch_perm = torch.randperm(len(self.branch_data_points))
            self.branch_data_points = self.branch_data_points[branch_perm]
            self.out_data_points = self.out_data_points[branch_perm]
            self.trunk_data_points = self.trunk_data_points[branch_perm]

        self.trunk_batch_size = len(self.trunk_data_points[0]) if trunk_batch_size < 0 else trunk_batch_size
        self.branch_batch_size = len(self.branch_data_points) if branch_batch_size < 0 else branch_batch_size

        self.branch_space = branch_space
        self.trunk_space = trunk_space
        self.output_space = output_space

        # for index computation in __getitem__ 
        self.branch_batch_len = int(np.ceil(len(self.branch_data_points) / self.branch_batch_size))
        self.trunk_batch_len = int(np.ceil(len(self.trunk_data_points[0]) / self.trunk_batch_size))
    
    def __len__(self):
        """Returns the number of points of this dataset.
        """
        # here we recompute, for the case when the batch size changed 
        self.branch_batch_len = int(np.ceil(len(self.branch_data_points) / self.branch_batch_size))
        self.trunk_batch_len = int(np.ceil(len(self.trunk_data_points[0]) / self.trunk_batch_size))
        return self.branch_batch_len * self.trunk_batch_len

    def __getitem__(self, idx):
        """Returns the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the desired point.
        """
        # frist slice in branch dimension (dim 0):
        branch_idx = int(idx / self.branch_batch_len)
        a = (branch_idx*self.branch_batch_size) % len(self.branch_data_points)
        b = ((branch_idx+1)*self.branch_batch_size) % len(self.branch_data_points)
        if a < b:
            branch_points = self.branch_data_points[a:b]
            out_points = self.out_data_points[a:b]
            trunk_points = self.trunk_data_points[a:b]
        else:
            branch_points = torch.cat([self.branch_data_points[a:], self.branch_data_points[:b]], dim=0)
            out_points = torch.cat([self.out_data_points[a:], self.out_data_points[:b]], dim=0)
            trunk_points = torch.cat([self.trunk_data_points[a:], self.trunk_data_points[:b]], dim=0)
        # then in trunk dimension (dim 1), only for trunk and output:
        trunk_idx = idx % self.trunk_batch_len
        a = (trunk_idx*self.trunk_batch_size) % len(self.trunk_data_points[0])
        b = ((trunk_idx+1)*self.trunk_batch_size) % len(self.trunk_data_points[0])
        if a < b:
            out_points = out_points[:, a:b, :]
            trunk_points = trunk_points[:, a:b, :]
        else:
            out_points = torch.cat([out_points[:, a:, :], out_points[:, :b, :]], dim=1)
            trunk_points = torch.cat([trunk_points[:, a:, :], trunk_points[:, :b, :]], dim=1)
        return (Points(branch_points, self.branch_space),
                Points(trunk_points, self.trunk_space),
                Points(out_points, self.output_space))


class DeepONetDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset to load tuples of data points, used via DeepONetDataLoader.
    Used if all branch inputs have the same trunk points.
    """
    def __init__(self, branch_data_points, trunk_data_points, out_data_points,
                 branch_space, trunk_space, output_space,
                 branch_batch_size, trunk_batch_size, shuffle_branch=False,
                 shuffle_trunk=True):

        assert out_data_points.shape[0] == branch_data_points.shape[0], \
            "Output values and branch inputs don't match!"
        assert out_data_points.shape[1] == trunk_data_points.shape[0], \
            "Output values and trunk inputs don't match!"

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