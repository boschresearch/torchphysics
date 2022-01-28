"""Contains a class that handles the storage of all created data points.
"""
from typing import Iterable
import torch

from .space import Space


class Points():
    """A set of points in a space, stored as a torch.Tensor.

    Parameters
    ----------
    data : torch.tensor, np.array or list
        The data points that should be stored.
        Have to be of the shape (batch_length, space.dimension).
    space : torchphysics.spaces.Space
        The space to which these points belongs to.

    Notes
    -----
    This class is essentially a combination of a torch.Tensor and a
    dictionary. So all data points can be stored as a single tensor, where
    we efficently can access and transform the data. But at the same time
    have the knowledge of what points belong to which space/variable.
    """
    def __init__(self, data, space, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.space = space
        assert len(self._t.shape) == 2
        assert self._t.shape[1] == self.space.dim
    
    @classmethod
    def empty(cls, **kwargs):
        """Creates an empty Points object.

        Returns
        -------
        Points
            The empty Points-object.
        """
        return cls(torch.empty(0,0, **kwargs), Space({}))
    
    @classmethod
    def joined(cls, *points_l):
        """Concatenates different Points to one single Points-Object.
        Will we use torch.cat on the data of the different Points and 
        create the product space of the Points spaces.

        Parameters
        ----------
        *points_l :
            The different Points that should be connected.

        Returns
        -------
        Points
            the created Points object.
        """
        points_out = []
        space_out = Space({})
        for points in points_l:
            if points.isempty:
                continue
            assert space_out.keys().isdisjoint(points.space)
            points_out.append(points._t)
            space_out = space_out * points.space
        return cls(torch.cat(points_out, dim=1), space_out)


    @classmethod
    def from_coordinates(cls, coords):
        """Concatenates sample coordinates from a dict to create a point
        object.

        Parameters
        ----------
        coords : dict
            The dictionary containing the data for every variable.

        Returns
        -------
        points : Points
            the created points object.
        """
        point_list = []
        space = {}
        if coords == {}:
            return cls.empty()
        n = coords[list(coords.keys())[0]].shape[0]
        for vname in coords:
            coords[vname] = torch.as_tensor(coords[vname])
            assert coords[vname].shape[0] == n
            point_list.append(coords[vname])
            space[vname] = coords[vname].shape[1]
        return cls(torch.column_stack(point_list), Space(space))
    
    @property
    def dim(self):
        """Returns the dimension of the points.
        """
        return self.space.dim
    
    @property
    def variables(self):
        """Returns variables of the points as an dictionary, e.g {'x': dim_x, 't': dim_t....}.
        """
        return self.space.variables
    
    @property
    def coordinates(self):
        """
        Returns a dict containing the coordinates of all points for each
        variable, e.g. {'x': torch.Tensor, 't': torch.Tensor}
        """
        out = {}
        variable_slice = self._variable_slices
        for var in self.space:
            out[var] = self._t[:, variable_slice[var]]
        return out
    
    @property
    def _variable_slices(self):
        start = 0
        slices = {}
        for v in self.space:
            stop = start + self.space[v]
            slices[v] = slice(start, stop, None)
            start += self.space[v]
        return slices

    @property
    def as_tensor(self):
        """Retunrs the underlying tensor.
        """
        return self._t
    
    def __len__(self):
        """Returns the number of points in this object.
        """
        return self._t.shape[0]
    
    @property
    def isempty(self):
        """Checks if no points are saved in this object.
        """
        return len(self) == 0 and self.space.dim == 0

    def __repr__(self):
        return "{}:\n{}".format(self.__class__.__name__, self.coordinates)
    
    def __getitem__(self, val):
        """
        Supports usual slice operations like points[1:3,('x','t')]. If a variable
        is given, this will return a torch.Tensor with the data. If not, it will
        return a new, sliced, point.

        Notes
        -----
        This operation does not support slicing single dimensions from a
        variable directly, however, this can be done on the output.
        """
        if not isinstance(val, tuple) and not isinstance(val, list):
            val = (val,)
        # first axis
        if isinstance(val[0], int):
            # keep tensor dimension
            out = self._t[val[0]:val[0]+1,:]
        else:
            out = self._t[val[0],:]
        out_space = self.space

        # second axis
        if len(val) == 2:
            slc = self._variable_slices
            rng = list(range(self.dim))
            out_idxs = []
            if val[1] in out_space:
                out_space = Space({val[1]: out_space[val[1]]})
            else:
                out_space = out_space[val[1]]
            for var in out_space:
                out_idxs += rng[slc[var]]
            out = out[:,out_idxs]

        return Points(out, out_space)
    
    def __iter__(self):
        """
        Iterates through points. It is in general not recommended
        to use this operation because it may lead to huge (and therefore
        slow) loops.
        """
        for i in range(len(self)):
            yield self[i, :]

    def __eq__(self, other):
        """Compares two Points if they are equal.
        """
        return self.space == other.space and torch.equal(self._t, other._t)
    
    def __add__(self, other):
        """Adds the data of two Points, have to lay in the same space.
        """
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t + other._t, self.space)

    def __sub__(self, other):
        """Substracts the data of two Points, have to lay in the same space.
        """
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t - other._t, self.space)

    def __mul__(self, other):
        """Pointwise multiplies the data of two Points, 
        have to lay in the same space.
        """
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t * other._t, self.space)

    def __pow__(self, other):
        """Pointwise raises the data of the first Points object to the
        power of the second one.
        """
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t ** other._t, self.space)

    def __truediv__(self, other):
        """Pointwise divides the data of two Points, 
        have to lay in the same space.
        """
        assert isinstance(other, Points)
        assert other.space == self.space
        return Points(self._t / other._t, self.space)
    
    def __or__(self, other):
        """Appends the data points of the second Point behind the 
        data of the first Point. (torch.cat((data_1, data_2), dim=0))
        """
        assert isinstance(other, Points)
        if self.isempty:
            return other
        if other.isempty:
            return self
        assert other.space == self.space
        return Points(torch.cat([self._t, other._t], dim=0), self.space)
    
    def join(self, other):
        """Stacks the data points of the second Point behind the 
        data of the first Point. (torch.cat((data_1, data_2), dim=1))
        """
        assert isinstance(other, Points)
        if self.isempty:
            return other
        if other.isempty:
            return self
        assert self.space.keys().isdisjoint(other.space)
        return Points(torch.cat([self._t, other._t], dim=1), self.space * other.space)

    def repeat(self, *sizes):
        """Repeats this points data along the specified dimensions. 
        Uses torch.repeat and will therefore repeat the data 'batchwise'.

        Parameters
        ----------
        *sizes :
            The number of repeats per dimension. 
        """
        return Points(self._t.repeat(*sizes), self.space)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        A helper method to create compatibility with most torch operations.
        Heavily inspired by the official torch documentation.
        """
        if kwargs is None:
            kwargs = {}
        args_list = [a._t if hasattr(a, '_t') else a for a in args]
        spaces = tuple(a.space for a in args if hasattr(a, 'space'))
        assert len(spaces) > 0
        ret = func(*args_list, **kwargs)
        return ret
    
    @property
    def requires_grad(self):
        """Returns the '.requires_grad' property of the underlying Tensor.
        """
        return self._t.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        """Sets the '.requires_grad' property of the underlying Tensor.

        Parameter
        ---------
        value : bool
            If gradients are required or not.
        """
        self._t.requires_grad = value
    
    def cuda(self, *args, **kwargs):
        self._t = self._t.cuda(*args, **kwargs)
        return self
    
    def to(self, *args, **kwargs):
        """Moves the underlying Tensor to other hardware parts.
        """
        self._t = self._t.to(*args, **kwargs)
        return self
