"""Contains a class that handles the storage of all created data points.
"""
from typing import Iterable
import torch
import numpy as np

from .space import Space


class Points():
    """A set of points in a space, stored as a torch.Tensor. Can contain
    multiple axis which keep batch-dimensions.

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
        assert len(self._t.shape) >= 2
        assert self._t.shape[-1] == self.space.dim

    @classmethod
    def empty(cls, **kwargs):
        """Creates an empty Points object.

        Returns
        -------
        Points
            The empty Points-object.
        """
        return cls(torch.empty(0, 0, **kwargs), Space({}))

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
        shape = points_l[0].shape
        for points in points_l:
            if points.isempty:
                continue
            assert space_out.keys().isdisjoint(points.space)
            assert points.shape == shape
            points_out.append(points._t)
            space_out = space_out * points.space
        return cls(torch.cat(points_out, dim=-1), space_out)

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
        n = coords[list(coords.keys())[0]].shape[:-1]
        for vname in coords:
            coords[vname] = torch.as_tensor(coords[vname])
            assert coords[vname].shape[:-1] == n
            point_list.append(coords[vname])
            space[vname] = coords[vname].shape[-1]
        return cls(torch.cat(point_list, dim=-1), Space(space))

    @property
    def dim(self):
        """Returns the dimension of the points.
        """
        return self.space.dim

    @property
    def variables(self):
        """Returns variables of the points as an unordered set, e.g {'x', 't'}.
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
            out[var] = self._t[..., variable_slice[var]]
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
        """
        Returns the underlying tensor.
        """
        return self._t

    @property
    def device(self):
        """
        Returns the device of the underlying tensor.
        """
        return self._t.device

    def __len__(self):
        """
        Returns the number of points in this object.
        """
        return np.prod(self._t.shape[:-1])

    @property
    def shape(self):
        """
        The shape of the batch-dimensions of this points object.
        """
        return self._t.shape[:-1]

    @property
    def isempty(self):
        """
        Checks whether no points and no structure are saved in this object.
        """
        return len(self) == 0 and self.space.dim == 0

    def __repr__(self):
        return "{}:\n{}".format(self.__class__.__name__, self.coordinates)

    def _compute_slice(self, val):
        if isinstance(val, tuple):
            val = list(val)
        
        if isinstance(val, (np.ndarray, torch.Tensor)) and val.dtype in (bool, torch.bool):
            if len(val.shape) == len(self._t.shape):
                raise IndexError("Boolean slicing in last dimension is not supported.")

        out_space = self.space
        if isinstance(val, list):
            # check if Ellipsis(...) is inside the slicing input.
            # Here we have to be carefull if specific indices are passed in, as an 
            # array/tensor since they do not allow to check: Ellipse in val
            # because then the check Ellipse == val[i] is used -> returns array
            slice_is_correct = True
            for slice_value in val:
                slice_is_correct = (slice_value is Ellipsis)
                if slice_is_correct:
                    break
            # check last element is not Ellipsis:
            slice_is_correct = (slice_is_correct and not val[-1] is Ellipsis)
            # compute slice structure
            if (len(val) == len(self._t.shape)) or slice_is_correct:
                slc = self._variable_slices
                rng = list(range(self.dim))
                if isinstance(val[-1], str):
                    out_space = Space({val[-1]: self.space[val[-1]]})
                    val[-1] = slc[val[-1]]
                else:
                    out_space = self.space[val[-1]]
                    out_idxs = []
                    for var in out_space:
                        out_idxs += rng[slc[var]]
                    val[-1] = out_idxs

        return val, out_space

    def __getitem__(self, val):
        """
        Supports usual slice operations like points[1:3,('x','t')]. Returns a new,
        sliced, points object.
        """
        val, space = self._compute_slice(val)
        out = self._t[val]
        if len(out.shape) == 1:
            out = out.unsqueeze(dim=0)
        return Points(out, space)

    def __setitem__(self, key, points):
        """
        Supports assignment of new point to the points tensor. Points are replaced
        using the same slicing rules as in `__setitem__`.
        """
        val, space = self._compute_slice(key)
        assert space == points.space
        self._t[val] = points._t

    def __iter__(self):
        """
        Iterates through first batch-dim. It is in general not recommended
        to use this operation because it may lead to huge (and therefore
        slow) loops.
        """
        for i in range(self._t.shape[0]):
            yield self[i]

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
        """Appends the data points of the second Points behind the 
        data of the first Points in the first batch-dim.
        (torch.cat((data_1, data_2), dim=0))
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
        data of the first Point. (torch.cat((data_1, data_2), dim=-1))
        """
        assert isinstance(other, Points)
        if self.isempty:
            return other
        if other.isempty:
            return self
        assert self.space.keys().isdisjoint(other.space)
        return Points(torch.cat([self._t, other._t], dim=-1), self.space * other.space)

    def repeat(self, *n):
        """Repeats this points data along the first batch-dimension. 
        Uses torch.repeat and will therefore repeat the data 'batchwise'.

        Parameters
        ----------
        n :
            The number of repeats. 
        """
        return Points(self._t.repeat(*n, *(((len(self._t.shape)-len(n)))*[1])), self.space)
    
    def unsqueeze(self, dim):
        """Adds an additional dimension inside the batch dimensions.

        Parameters
        ----------
        dim :
            Where to add the additional axis (considered only inside batch dimensions).
        """
        assert dim < len(self._t.shape)
        if dim < 0:
            dim -= 1
        return Points(self._t.unsqueeze(dim=dim), self.space)

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
    
    def track_coord_gradients(self):
        points_coordinates = self.coordinates
        for var in points_coordinates:
            points_coordinates[var].requires_grad = True
        return points_coordinates, Points.from_coordinates(points_coordinates)
