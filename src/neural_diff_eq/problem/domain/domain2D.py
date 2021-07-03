import numpy as np
import shapely.geometry as s_geo
from shapely.ops import triangulate

from .domain import Domain


class Rectangle(Domain):
    '''Class for arbitrary rectangles in 2D.

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
        The error tolerance for checking if points are inside or at the boundary.
    '''

    def __init__(self, corner_dl, corner_dr, corner_tl, tol=1e-06):
        self._check_rectangle(corner_dl, corner_dr, corner_tl, tol)
        self.corner_dl = np.asarray(corner_dl)
        self.corner_dr = np.asarray(corner_dr)
        self.corner_tl = np.asarray(corner_tl)
        self.length_lr = np.linalg.norm(self.corner_dr-self.corner_dl)
        self.length_td = np.linalg.norm(self.corner_tl-self.corner_dl)
        self.normal_lr = (self.corner_dr-self.corner_dl)/self.length_lr
        self.normal_td = (self.corner_tl-self.corner_dl)/self.length_td
        super().__init__(dim=2, volume=self.length_lr*self.length_td,
                         surface=2*(self.length_lr+self.length_td), tol=tol)
        # inverse matrix to transform the rectangle back to the unit square. Used
        # to check if points are inside or on the boundary of the rectangle
        self.inverse_matrix = [self.normal_lr/self.length_lr,
                               self.normal_td/self.length_td]

    def _check_rectangle(self, corner_1, corner_2, corner_3, tol):
        dot_prod = np.dot(np.array(corner_2)-np.array(corner_1),
                          np.array(corner_3)-np.array(corner_1))
        if not np.isclose(dot_prod, 0, atol=tol):
            raise ValueError('Input is not a rectangle')
        return

    def _transform_to_unit_square(self, points, corner):
        return np.array([np.matmul(self.inverse_matrix,
                                   np.subtract(i, corner)) for i in points])

    def _is_inside_unit_square(self, points):
        return ((points[:, 0] > -self.tol) & (points[:, 0] < 1+self.tol)
                & (points[:, 1] > -self.tol) & (points[:, 1] < 1+self.tol))

    def is_inside(self, points):
        '''Checks if the given points are inside the open rectangle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside,
            or false if not.
        '''
        transform = self._transform_to_unit_square(points, self.corner_dl)
        return self._is_inside_unit_square(transform).reshape(-1, 1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the rectangle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        transform = self._transform_to_unit_square(points, self.corner_dl)
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
        # append random points if there are not enough points in the grid
        if len(points) < n:
            points = np.append(points, self._random_sampling_inside(n-len(points)),
                               axis=0)
        return points.astype(np.float32)

    def _transform_to_rectangle(self, points):
        trans_matrix = np.column_stack(
            (self.corner_dr-self.corner_dl, self.corner_tl-self.corner_dl))
        points = [np.matmul(trans_matrix, p) for p in points]
        return np.add(points, self.corner_dl)

    def _random_sampling_boundary(self, n):
        n_x = int(np.floor(self.length_lr*n/(self.length_lr+self.length_td)))
        n_y = int(np.ceil(self.length_td*n/(self.length_lr+self.length_td)))
        side_td = self._construct_random_boundary_sides(n_x,
                                                        self.corner_dr-self.corner_dl,
                                                        self.corner_tl-self.corner_dl)
        side_lr = self._construct_random_boundary_sides(n_y,
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
        line_points = np.linspace(0, self.surface, n+1)[:-1]
        corners = [self.corner_dl, self.corner_dr,
                   self.corner_dr + (self.corner_tl-self.corner_dl),
                   self.corner_tl, self.corner_dl]
        side_lengths = [self.length_lr, self.length_td,
                        self.length_lr, self.length_td]
        points = np.empty((0, self.dim))
        for i in range(len(line_points)):
            for k in range(len(corners)-1):
                if line_points[i] < sum(side_lengths[:k+1]):
                    norm = side_lengths[k]
                    coord = line_points[i] - sum(side_lengths[:k])
                    new_point = corners[k] + coord/norm * (corners[k+1]-corners[k])
                    points = np.append(points, [new_point], axis=0)
                    break
        return points.astype(np.float32)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
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

        transform = self._transform_to_unit_square(points, self.corner_dl)
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

    def grid_for_plots(self, n):
        nx = int(np.ceil(np.sqrt(n)*self.length_lr/self.length_td))
        ny = int(np.ceil(np.sqrt(n)*self.length_td/self.length_lr))
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        return self._transform_to_rectangle(points).astype(np.float32)

    def serialize(self):
        # For showing data/information in tensorbaord
        dct = super().serialize()
        dct['name'] = 'Rectangle'
        dct['corner_dl'] = [int(a) for a in list(self.corner_dl)]
        dct['corner_dr'] = [int(a) for a in list(self.corner_dr)]
        dct['corner_tl'] = [int(a) for a in list(self.corner_tl)]
        return dct

    def _compute_bounds(self):
        corners = np.array([self.corner_dl, self.corner_dr, self.corner_tl,
                            self.corner_tl-self.corner_dl+self.corner_dr])
        min_x = np.min(corners[:, :1])
        max_x = np.max(corners[:, :1])
        min_y = np.min(corners[:, 1:])
        max_y = np.max(corners[:, 1:])
        return [min_x, max_x, min_y, max_y]


class Circle(Domain):
    '''Class for arbitrary circles.

    Parameters
    ----------
    center : array_like
        The center of the circle, e.g. center = [5,0].
    radius : number
        The radius of the circle.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    '''

    def __init__(self, center, radius, tol=1e-06):
        super().__init__(dim=2, volume=np.pi*radius**2,
                         surface=2*np.pi*radius, tol=tol)
        self.center = np.asarray(center)
        self.radius = radius

    def is_inside(self, points):
        '''Checks if the given points are inside the open circle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside, or
            false if not.
        '''
        points = np.subtract(points, self.center)
        return (np.linalg.norm(points, axis=1)[:] < self.radius+self.tol)\
            .reshape(-1, 1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the circle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

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
        # append random points if there are not enough points in the grid
        if len(points) < n:
            points = np.append(points, self._random_sampling_inside(n-len(points)),
                               axis=0)
        if len(points) > n:
            points = super()._cut_points(points, n)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        phi = 2 * np.pi * np.random.uniform(0, 1, n).reshape(-1, 1)
        points = np.column_stack((self.radius*np.cos(phi), self.radius*np.sin(phi)))
        return np.add(self.center, points).astype(np.float32)

    def _grid_sampling_boundary(self, n):
        phi = 2 * np.pi * np.linspace(0, 1, n+1)[0:-1].reshape(-1, 1)
        points = np.column_stack((self.radius*np.cos(phi), self.radius*np.sin(phi)))
        return np.add(self.center, points).astype(np.float32)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
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
        normal_vectors = np.subtract(points, self.center) / self.radius
        return normal_vectors

    def grid_for_plots(self, n):
        points_inside = self._grid_sampling_inside(n)
        # add some points at the boundary to better show the form of the circle
        points_boundary = self._grid_sampling_boundary(int(np.ceil(n/4)))
        points = np.append(points_inside, points_boundary, axis=0)
        return points.astype(np.float32)

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct['name'] = 'Circle'
        dct['center'] = [int(a) for a in list(self.center)]
        dct['radius'] = self.radius
        return dct

    def _compute_bounds(self):
        min_x = self.center[0] - self.radius
        max_x = self.center[0] + self.radius
        min_y = self.center[1] - self.radius
        max_y = self.center[1] + self.radius
        return [min_x, max_x, min_y, max_y]


class Triangle(Domain):
    '''Class for triangles in 2D.

    Parameters
    ----------
    corner_1, corner_2, corner_3 : array_like
        The three corners of the triangle.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    '''

    def __init__(self, corner_1, corner_2, corner_3, tol=1e-06):
        self.corners = np.array([corner_1, corner_2, corner_3, corner_1])
        volume = self._compute_area()
        self.side_lengths = self._compute_side_lengths()
        self.normals = self._compute_normals()
        self.inverse_matrix = self._compute_inverse()
        super().__init__(dim=2, volume=volume,
                         surface=sum(self.side_lengths), tol=tol)

    def _compute_area(self):
        area = 1/2 * np.cross(self.corners[1]-self.corners[0],
                              self.corners[2]-self.corners[0])
        if area < 0:
            self.corners = self.corners[::-1]
            area = -area
        return area

    def _compute_side_lengths(self):
        # Function is also used by the Polygon2D class
        side_length = np.zeros(len(self.corners)-1)
        for i in range(len(self.corners)-1):
            side_length[i] = np.linalg.norm(self.corners[i+1]-self.corners[i])
        return side_length

    def _compute_normals(self):
        # Function is also used by the Polygon2D class
        normals = np.zeros((len(self.corners)-1, 2))
        for i in range(len(self.corners)-1):
            normals[i] = np.subtract(self.corners[i+1], self.corners[i])[::-1]
            normals[i][1] *= -1
            normals[i] /= self.side_lengths[i]
        return normals

    def _compute_inverse(self):
        matrix = np.column_stack((self.corners[1]-self.corners[0],
                                  self.corners[2]-self.corners[0]))
        return np.linalg.inv(matrix)

    def is_inside(self, points):
        '''Checks if the given points are inside the triangle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside,
            or false if not.
        '''
        transform = Rectangle._transform_to_unit_square(self, points, self.corners[0])
        return self._is_inside_unit_triangle(transform).reshape(-1, 1)

    def _is_inside_unit_triangle(self, points):
        inside_x = ((points[:, 0] > -self.tol) & (points[:, 0] < 1+self.tol))
        inside_y = ((points[:, 1] > -self.tol) & (points[:, 1] < 1+self.tol))
        bary_smaller_1 = (points.sum(axis=1) < 1+self.tol)
        return np.logical_and(inside_x, np.logical_and(inside_y, bary_smaller_1))

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the triangle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...]

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        return self._where_on_boundary(points)

    def _where_on_boundary(self, points, return_index=False):
        transform = Rectangle._transform_to_unit_square(self, points, self.corners[0])
        inside = self._is_inside_unit_triangle(transform)
        on_x = np.isclose(transform[:, 0], 0, atol=self.tol)
        on_y = np.isclose(transform[:, 1], 0, atol=self.tol)
        on_xy = np.isclose(transform.sum(axis=1), 1, atol=self.tol)
        on_bound = np.logical_or(on_x, np.logical_or(on_y, on_xy))
        on_bound = np.logical_and(inside, on_bound)
        if return_index:
            index = np.zeros((len(points), 1), dtype=int)
            index[np.where(on_x)[0]] = 2
            index[np.where(on_xy)[0]] = 1
            index[np.where(on_y)[0]] = 0
            return on_bound, index
        return on_bound.reshape(-1, 1)

    def _random_sampling_inside(self, n):
        return Triangle._random_points_in_triangle(n, self.corners)

    def _random_points_in_triangle(n, corners):
        bary_coords = np.random.uniform(0, 1, (n, 2))
        # if a barycentric coordinates is bigger then 1, mirror them at the
        # point (0.5, 0.5). Stays uniform.
        index = np.where(bary_coords.sum(axis=1) > 1)[0]
        bary_coords[index] = np.subtract([1, 1], bary_coords[index])
        axis_1 = np.multiply(corners[1]-corners[0], bary_coords[:, :1])
        axis_2 = np.multiply(corners[2]-corners[0], bary_coords[:, 1:])
        return np.add(np.add(corners[0], axis_1), axis_2).astype(np.float32)

    def _grid_sampling_inside(self, n):
        bounds = self._compute_bounds()
        return self._grid_sampling_with_bbox(n, bounds)

    def _compute_bounds(self):
        min_x = np.min(self.corners[:, :1])
        max_x = np.max(self.corners[:, :1])
        min_y = np.min(self.corners[:, 1:])
        max_y = np.max(self.corners[:, 1:])
        return [min_x, max_x, min_y, max_y]

    def _grid_sampling_with_bbox(self, n, bounds):
        bounding_box = Rectangle([bounds[0], bounds[2]], [bounds[1], bounds[2]],
                                 [bounds[0], bounds[3]])
        scaled_n = int(bounding_box.volume/self.volume * n)
        points = bounding_box._grid_sampling_inside(scaled_n)
        inside = self.is_inside(points)
        index = np.where(np.invert(inside))[0]
        points = np.delete(points, index, axis=0)
        if len(points) < n:
            points = np.append(points, self._random_sampling_inside(n-len(points)),
                               axis=0)
        if len(points) > n:
            points = self._cut_points(points, n)
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        line_points = np.random.uniform(0, self.surface, n)
        return self._distribute_line_to_boundary(line_points)

    def _grid_sampling_boundary(self, n):
        line_points = np.linspace(0, self.surface, n+1)[:-1]
        return self._distribute_line_to_boundary(line_points)

    def _distribute_line_to_boundary(self, line_points):
        points = np.empty((0, 2))
        for i in range(len(line_points)):
            for k in range(len(self.corners)-1):
                if line_points[i] < sum(self.side_lengths[:k+1]):
                    norm = self.side_lengths[k]
                    coord = line_points[i] - sum(self.side_lengths[:k])
                    new_point = (self.corners[k] + coord/norm *
                                 (self.corners[k+1]-self.corners[k]))
                    points = np.append(points, [new_point], axis=0)
                    break
        return points.astype(np.float32)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector has to be computed,e.g.
            [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        on_bound, index = self._where_on_boundary(points, return_index=True)
        if not all(on_bound):
            print('Warning: some points are not at the boundary!')
        normals = np.zeros((len(points), self.dim))
        for i in range(len(points)):
            normals[i] = self.normals[index[i]]
        return normals.astype(np.float32)

    def grid_for_plots(self, n):
        points_inside = self._grid_sampling_inside(n)
        # add some points at the boundary to better show the form of the circle
        points_boundary = self._grid_sampling_boundary(int(np.ceil(n/4)))
        return np.append(points_inside, points_boundary, axis=0).astype(np.float32)

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct['name'] = 'Triangle'
        dct['corner_1'] = [int(a) for a in list(self.corners[0])]
        dct['corner_2'] = [int(a) for a in list(self.corners[1])]
        dct['corner_3'] = [int(a) for a in list(self.corners[2])]
        return dct

class Polygon2D(Domain):
    '''Class for polygons in 2D.

    Parameters
    ----------
    corners : list of lists, optional 
        The corners/vertices of the polygon. Can be eihter in clockwise or counter-
        clockwise order. 
    shapely_polygon : shapely.geometry.Polygon, optional
        Instead of defining the corner points, it is also possible to give a already
        existing shapely.Polygon object.  
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    '''

    def __init__(self, corners=None, shapely_polygon=None, tol=1e-06):
        if corners is not None:
            self._check_not_triangle(corners)
            self.polygon = s_geo.Polygon(corners)
        elif isinstance(shapely_polygon, s_geo.Polygon):
            self.polygon = shapely_polygon
        else:
            raise ValueError('Needs either points to create a new'
                             'polygon, or a existing shapely polygon.')
        self.polygon = s_geo.polygon.orient(self.polygon)
        super().__init__(dim=2, tol=tol, volume=self.polygon.area,
                         surface=self.polygon.boundary.length)
        self.corners = np.array(self.polygon.exterior.coords)
        self.side_lengths = Triangle._compute_side_lengths(self)
        self.normals = Triangle._compute_normals(self)

    def _check_not_triangle(self, points):
        if len(points) == 3:
            raise ValueError('It is more efficient to use the triangle class!')

    def is_inside(self, points):
        '''Checks if the given points are inside the polygon.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was inside,
            or false if not.
        '''
        inside = np.empty(len(points), dtype=bool)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            inside[i] = self.polygon.contains(point)
        return inside.reshape(-1, 1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the boundary of the polygon.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked. The list has to be of
            the form [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true, if the points was on the
            boundary, or false if not.
        '''
        on_bound = np.empty(len(points), dtype=bool)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            distance = self.polygon.boundary.distance(point)
            on_bound[i] = (np.abs(distance) <= self.tol)
        return on_bound.reshape(-1, 1)

    def grid_for_plots(self, n):
        return Triangle.grid_for_plots(self, n)

    def _random_sampling_inside(self, n):
        points = np.empty((0, self.dim))
        big_t, t_area = None, 0
        for t in triangulate(self.polygon):
            new_points = self._sample_in_triangulation(t, n)
            points = np.append(points, new_points, axis=0)
            if t.within(self.polygon) and t.area > t_area:
                big_t = [t][0]
                t_area = t.area
        if len(points) < n:
            points = np.append(points,
                               self._sample_in_triangulation(big_t, n-len(points)),
                               axis=0)
        if len(points) > n:
            points = self._cut_points(points, n)
        return points.astype(np.float32)

    def _sample_in_triangulation(self, t, n):
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        corners = np.array([[x0, y0], [x1, y1], [x2, y2]])
        scaled_n = int(np.ceil(t.area/self.volume * n))
        new_points = Triangle._random_points_in_triangle(scaled_n, corners)
        if not t.within(self.polygon):
            inside = self.is_inside(new_points)
            index = np.where(inside)[0]
            new_points = new_points[index]
        return new_points

    def _grid_sampling_inside(self, n):
        bounds = self._compute_bounds()
        return Triangle._grid_sampling_with_bbox(self, n, bounds)

    def _random_sampling_boundary(self, n):
        line_points = np.random.uniform(0, self.surface, n)
        return Triangle._distribute_line_to_boundary(self, line_points)

    def _grid_sampling_boundary(self, n):
        line_points = np.linspace(0, self.surface, n+1)[:-1]
        return Triangle._distribute_line_to_boundary(self, line_points)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector has to be computed,e.g.
            [[x1,y1],[x2,y2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point
            specified in the input array.
        '''
        index = self._where_on_boundary(points)
        if (index == -1).any():
            print('Warning: some points are not at the boundary!')
        normals = np.zeros((len(points), self.dim))
        for i in range(len(points)):
            normals[i] = self.normals[index[i]]
        return normals.astype(np.float32)

    def _where_on_boundary(self, points):
        index = -1 * np.ones(len(points), dtype=int)
        coords = self.polygon.exterior.coords
        for i in range(len(self.corners)-1):
            line = s_geo.LineString([coords[i], coords[i+1]])
            for k in np.where(index < 0)[0]:
                point = s_geo.Point(points[k])
                distance = line.distance(point)
                if np.abs(distance) <= self.tol:
                    index[k] = i
        return index

    def _compute_bounds(self):
        bounds = self.polygon.bounds
        return [bounds[0], bounds[2], bounds[1], bounds[3]]

    def serialize(self):
        # to show data/information in tensorboard
        dct = super().serialize()
        dct['name'] = 'Polygon2D'
        for i in range(len(self.corners)-1):
            dct['corner_' + str(i)] = [int(a) for a in list(self.corners[i])]
        return dct