import numpy as np

from .domain import Domain


class Rectangle(Domain):
    '''Class for arbitrary rectangles in 2D

    Parameters
    ----------
    corner_dl, corner_dr, corner_tl : array_like
        Three corners of the rectangle, in the following order
            tl ----- x
            |        |
            |        |
            dl ----- dr
        (dl = down left, dr = down right, tl = top left)
        E.g. for the unit square: corner_dl = [0,0], corner_dr = [1,0],
                                  corner_tl = [0,1].
    tol : number, optional
        The error toleranz for checking if points are inside or at the boundary
    '''

    def __init__(self, corner_dl, corner_dr, corner_tl, tol=1e-06):
        super().__init__(dim=2, tol=tol)
        self.corner_dl = np.asarray(corner_dl)
        self.corner_dr = np.asarray(corner_dr)
        self.corner_tl = np.asarray(corner_tl)
        self._check_rectangle()
        self.length_lr = np.linalg.norm(self.corner_dr-self.corner_dl)
        self.length_td = np.linalg.norm(self.corner_tl-self.corner_dl)
        self.normal_lr = (self.corner_dr-self.corner_dl)/self.length_lr
        self.normal_td = (self.corner_tl-self.corner_dl)/self.length_td
        # inverse matrix to transform the rectangle back to the unit square. Used
        # to check if points are inside or on the boundary of the rectangle
        self.inverse_matrix = [self.normal_lr /
                               self.length_lr, self.normal_td/self.length_td]

    def _check_rectangle(self):
        dot_prod = np.dot(self.corner_dr-self.corner_dl, self.corner_tl-self.corner_dl)
        if not np.isclose(dot_prod, 0, atol=self.tol):
            raise ValueError('Input is not a rectangle')
        return

    def _transform_to_unit_square(self, points):
        return np.array([np.matmul(self.inverse_matrix,
                                   np.subtract(i, self.corner_dl)) for i in points])

    def _is_inside_unit_square(self, points):
        return ((points[:, 0] >= -self.tol) & (points[:, 0] <= 1+self.tol)
                & (points[:, 1] >= -self.tol) & (points[:, 1] <= 1+self.tol))

    def is_inside(self, points):
        '''Checks if the given points are inside the rectangle

        Parameters
        ----------
        points : array_like
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside, or
            false if not.
        '''
        transform = self._transform_to_unit_square(points)
        return self._is_inside_unit_square(transform).reshape(-1, 1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the rectangle

        Parameters
        ----------
        points : array_like
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        transform = self._transform_to_unit_square(points)
        inside = self._is_inside_unit_square(transform)
        boundary = ((np.isclose(transform[:, 0], 0, atol=self.tol))
                    | (np.isclose(transform[:, 0], 1, atol=self.tol))
                    | (np.isclose(transform[:, 1], 0, atol=self.tol))
                    | (np.isclose(transform[:, 1], 1, atol=self.tol)))
        return np.logical_and(inside, boundary).reshape(-1, 1)

    def _random_sampling_inside(self, n):
        axis_1 = self._sampling_axis(n, self.corner_dr-self.corner_dl)
        axis_2 = self._sampling_axis(n, self.corner_tl-self.corner_dl)
        return np.add(np.add(self.corner_dl, axis_1), axis_2).astype(np.float32)

    def _sampling_axis(self, n, direction):
        points = [np.random.uniform(0, 1) for i in range(n)]
        return [(t*direction) for t in points]

    def _grid_sampling_inside(self, n):
        nx = int(np.sqrt(n*self.length_lr/self.length_td))
        ny = int(np.sqrt(n*self.length_td/self.length_lr))
        x = np.linspace(0, 1, nx+2)[1:-1]
        y = np.linspace(0, 1, ny+2)[1:-1]
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        trans_matrix = np.column_stack(
            (self.corner_dr-self.corner_dl, self.corner_tl-self.corner_dl))
        points = [np.matmul(trans_matrix, p) for p in points]
        points = np.add(points, self.corner_dl)
        # append the center if there are not enough points in the grid
        while len(points) < n:
            points = np.append(points, [1/2.0*(self.corner_dr+self.corner_tl)], axis=0)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        nx, ny = self._divide_boundary_points(n)
        side_td = self._construct_random_boundary_sides(nx,
                                                        self.corner_dr-self.corner_dl,
                                                        self.corner_tl-self.corner_dl)
        side_lr = self._construct_random_boundary_sides(ny,
                                                        self.corner_tl-self.corner_dl,
                                                        self.corner_dr-self.corner_dl)
        return np.concatenate((side_td, side_lr)).astype(np.float32)

    def _construct_random_boundary_sides(self, n, direction, replacement):
        if n > 0:
            sides = np.add(self.corner_dl, self._sampling_axis(n, direction))
            rand_int = np.random.randint(0, 2, n).reshape(-1, 1)
            sides = np.add(sides, (replacement) * rand_int)
            return sides
        return np.empty((0, 2))

    def _grid_sampling_boundary(self, n):
        nx, ny = self._divide_boundary_points(n)
        corner_tr = self.corner_dr + self.corner_tl - self.corner_dl
        axis_1 = np.linspace(self.corner_dl, self.corner_dr, int(np.ceil(nx/2))+1)[0:-1]
        axis_2 = np.linspace(self.corner_dr, corner_tr, int(np.ceil(ny/2))+1)[0:-1]
        axis_3 = np.linspace(corner_tr, self.corner_tl, int(np.floor(nx/2))+1)[0:-1]
        axis_4 = np.linspace(self.corner_tl, self.corner_dl,
                             int(np.floor(ny/2))+1)[0:-1]
        points = np.concatenate((axis_1, axis_2, axis_3, axis_4))
        if len(points) < n:
            points = np.append(points, [self.corner_dl], axis=0)
        return points.astype(np.float32)

    def _divide_boundary_points(self, n):
        n1 = int(np.floor(self.length_lr*n/(self.length_lr+self.length_td)))
        n2 = int(np.ceil(self.length_td*n/(self.length_lr+self.length_td)))
        return n1, n2

    def boundary_normal(self, points):
        '''Computes the boundary normal

        Parameters
        ----------
        points : array_like
            A list containing all points where the normal vector has to be computed,e.g.
            [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        if not all(self.is_on_boundary(points)):
            print('Warning: some points are not at the boundary!')

        transform = self._transform_to_unit_square(points)
        left = np.isclose(transform[:, 0], 0, atol=self.tol).reshape(-1, 1)
        right = np.isclose(transform[:, 0], 1, atol=self.tol).reshape(-1, 1)
        down = np.isclose(transform[:, 1], 0, atol=self.tol).reshape(-1, 1)
        top = np.isclose(transform[:, 1], 1, atol=self.tol).reshape(-1, 1)
        normal_vectors = (self.normal_lr*right - self.normal_lr*left
                          - self.normal_td*down + self.normal_td*top)
        # check if there is a corner point:
        index_tl = np.where(np.logical_and(top, left))
        index_tr = np.where(np.logical_and(top, right))
        index_dl = np.where(np.logical_and(down, left))
        index_dr = np.where(np.logical_and(down, right))
        index = np.concatenate((index_dl[0], index_tl[0], index_dr[0], index_tr[0]))
        # rescale the vector in the corner
        normal_vectors[index] *= 1/np.sqrt(2)
        return normal_vectors


class Circle(Domain):
    '''Class for arbitrary circles

    Parameters
    ----------
    center : array_like
        The center of the circle, e.g. center = [5,0]
    radius : number
        The radius of the circle
    tol : number, optional
        The error toleranz for checking if points are inside or at the boundary
    '''

    def __init__(self, center, radius, tol=1e-06):
        super().__init__(dim=2, tol=tol)
        self.center = np.asarray(center)
        self.radius = radius

    def is_inside(self, points):
        '''Checks if the given points are inside the circle

        Parameters
        ----------
        points : array_like
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside, or
            false if not.
        '''
        points = np.subtract(points, self.center)
        return (np.linalg.norm(points, axis=1)[:] <= self.radius+self.tol)\
            .reshape(-1, 1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the circle

        Parameters
        ----------
        points : array_like
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        norm = np.linalg.norm(np.subtract(points, self.center), axis=1)
        return (np.isclose(norm[:], self.radius, atol=self.tol)).reshape(-1, 1)

    def _random_sampling_inside(self, n):
        r = self.radius * np.sqrt(np.random.uniform(0, 1, n)).reshape(-1, 1)
        phi = 2 * np.pi * np.random.uniform(0, 1, n).reshape(-1, 1)
        points = np.column_stack(
            (np.multiply(r, np.cos(phi)), np.multiply(r, np.sin(phi))))
        return np.add(self.center, points).astype(np.float32)

    def _grid_sampling_inside(self, n):
        scaled_n = 2*int(np.sqrt(n/np.pi))
        axis = np.linspace(-self.radius, self.radius, scaled_n+2)[1:-1]
        points = np.array(np.meshgrid(axis, axis)).T.reshape(-1, 2)
        points = np.add(points, self.center)
        inside = np.nonzero(self.is_inside(points))[0]
        points = points[inside]
        # append the center if there are not enough points in the grid
        while len(points) < n:
            points = np.append(points, [self.center], axis=0)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        phi = 2 * np.pi * np.random.uniform(0, 1, n).reshape(-1, 1)
        points = np.column_stack((self.radius*np.cos(phi), self.radius*np.sin(phi)))
        return np.add(self.center, points).astype(np.float32)

    def _grid_sampling_boundary(self, n):
        phi = 2 * np.pi * np.linspace(0, 1, n+1)[0:-1].reshape(-1, 1)
        points = np.column_stack((self.radius*np.cos(phi), self.radius*np.sin(phi)))
        return np.add(self.center, points).astype(np.float32)

    def boundary_normal(self, x):
        '''Computes the boundary normal

        Parameters
        ----------
        points : array_like
            A list containing all points where the normal vector has to be computed,e.g.
            [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        if not all(self.is_on_boundary(x)):
            print('Warninig: some points are not at the boundary!')
        normal_vectors = np.subtract(x, self.center) / self.radius
        return normal_vectors
