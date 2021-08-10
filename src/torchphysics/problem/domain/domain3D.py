import numpy as np
import trimesh
import logging

from .domain import Domain
from .domain2D import Circle, Polygon2D


class Box(Domain):
    '''Class for arbitrary boxes in 3D.

    Parameters
    ----------
    corner_o, corner_x, corner_y, corner_z : array_like
        Four corners of the Box, in the following form (or rotated)
            |      . ----- .
            |     / |     /|
            |   z ----- .  |
            |   |   |   |  |
            |   |  y ---|--.
            |   | /     | /
            |   o ----- x
        (o = corner origin, x = corner in "x" direction,
         y = corner in "y" direction, z = corner in "z" direction)
        E.g.: corner_oc = [0,0,0], corner_xc = [2,0,0],
              corner_yc = [0,1,0], corner_zc = [0,0,3].
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    '''
    def __init__(self, corner_o, corner_x, corner_y, corner_z, tol=1e-06):
        self.corner_o = np.array(corner_o)
        self.corner_x = np.array(corner_x)
        self.corner_y = np.array(corner_y)
        self.corner_z = np.array(corner_z)
        self._check_input_is_box(tol)
        self.side_lengths = self._compute_side_lengths()
        self.side_areas = self._compute_side_areas()
        super().__init__(dim=3, volume=np.prod(self.side_lengths),
                         surface=2*(np.sum(self.side_areas)),
                         tol=tol)
        self.normals = self._compute_normals()
        # inverse matrix to transform the box back to the unit cube:
        self.inverse_matrix = [self.normals[0]/self.side_lengths[0], 
                               self.normals[2]/self.side_lengths[1], 
                               self.normals[4]/self.side_lengths[2]]

    def _check_input_is_box(self, tol):
        # to check if we have a correct box compute the three angles
        prod_1 = np.dot(self.corner_x-self.corner_o,
                        self.corner_y-self.corner_o)
        prod_2 = np.dot(self.corner_x-self.corner_o,
                        self.corner_z-self.corner_o)
        prod_3 = np.dot(self.corner_z-self.corner_o,
                        self.corner_y-self.corner_o) 
        if not np.allclose([prod_1, prod_2, prod_3], 0, atol=tol):
            raise ValueError('Input is not a Box!')
        return

    def _compute_side_lengths(self):
        """Computes the vertice lengths of the box. 

        Returns
        -------
        list:
            The length in the form:
            [len(corner_x-corner_o), len(corner_y-corner_o), len(corner_z-corner_o)] 
        """
        side_1 = np.linalg.norm(self.corner_x-self.corner_o)
        side_2 = np.linalg.norm(self.corner_y-self.corner_o)
        side_3 = np.linalg.norm(self.corner_z-self.corner_o)
        return [side_1, side_2, side_3]

    def _compute_side_areas(self):
        """Computes the side areas of the box. 

        Returns
        -------
        list:
            The areas in the form:
            [x-y-areas, x-z-area, y-z-area] 
        """
        area_1 = self.side_lengths[0]*self.side_lengths[1]
        area_2 = self.side_lengths[0]*self.side_lengths[2]
        area_3 = self.side_lengths[2]*self.side_lengths[1]
        return [area_1, area_2, area_3]

    def _compute_normals(self):
        """Computes the normal vectors of the box. 

        Returns
        -------
        np.array:
            The normal vectors in the form:
            [x-normal, -x-normal, y-normal, -y-normal, ...] 
        """
        x_normal = (self.corner_x-self.corner_o)/self.side_lengths[0] 
        y_normal = (self.corner_y-self.corner_o)/self.side_lengths[1] 
        z_normal = (self.corner_z-self.corner_o)/self.side_lengths[2]
        return np.array([x_normal, -x_normal, y_normal, -y_normal, 
                         z_normal, -z_normal]).astype(np.float32)  

    def _transform_to_unit_cube(self, points):
        return np.array([np.matmul(self.inverse_matrix, np.subtract(i, self.corner_o))
                         for i in points])

    def _transform_unit_cube_to_box(self, points):
        trans_matrix = np.column_stack(
            (self.corner_x-self.corner_o, self.corner_y-self.corner_o, 
             self.corner_z-self.corner_o))
        points = [np.matmul(trans_matrix, p) for p in points]
        return np.add(points, self.corner_o)

    def _check_is_inside_unit_cube(self, points):
        in_x = (points[:, 0] > -self.tol) & (points[:, 0] < 1+self.tol)
        in_y = (points[:, 1] > -self.tol) & (points[:, 1] < 1+self.tol)
        in_z = (points[:, 2] > -self.tol) & (points[:, 2] < 1+self.tol)
        return np.logical_and(in_x, np.logical_and(in_y, in_z))

    def is_inside(self, points):
        """Checks if the given points are inside the box.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was inside, or false if not.
        """        
        points = self._transform_to_unit_cube(points)
        return self._check_is_inside_unit_cube(points).reshape(-1, 1)

    def is_on_boundary(self, points):
        """Checks if the given points are on the boundary of the circle.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was on the boundary, or false if not.
        """
        points = self._transform_to_unit_cube(points)
        inside = self._check_is_inside_unit_cube(points)
        bound_x = (np.isclose(points[:, 0], 0, atol=self.tol)
                   | np.isclose(points[:, 0], 1, atol=self.tol))
        bound_y = (np.isclose(points[:, 1], 0, atol=self.tol)
                   | np.isclose(points[:, 1], 1, atol=self.tol)) 
        bound_z = (np.isclose(points[:, 2], 0, atol=self.tol)
                   | np.isclose(points[:, 2], 1, atol=self.tol))  
        bound = np.logical_or(bound_x, np.logical_or(bound_y, bound_z))
        return np.logical_and(inside, bound).reshape(-1, 1)

    def _random_sampling_inside(self, n):
        x_axis = np.random.uniform(0, 1, (n,1))
        y_axis = np.random.uniform(0, 1, (n,1))
        z_axis = np.random.uniform(0, 1, (n,1))
        x_vector = (self.corner_x-self.corner_o) * x_axis
        y_vector = (self.corner_y-self.corner_o) * y_axis
        z_vector = (self.corner_z-self.corner_o) * z_axis
        points = x_vector + y_vector + z_vector
        return np.add(points, self.corner_o).astype(np.float32)

    def _grid_sampling_inside(self, n):
        points = self._grid_in_box(n)
        points = super()._check_inside_grid_enough_points(n, points)
        return points.astype(np.float32)

    def _grid_in_box(self, n):
        # divide number of points, depending on side length
        length_prod = np.prod(self.side_lengths)
        n_vec = np.zeros(3, dtype=int)
        for i in range(3):
            n_vec[i] = np.min([int(np.cbrt(n/length_prod)*self.side_lengths[i]), n])
        # create grid in unit cube
        x = np.linspace(0, 1, np.max([3, n_vec[0]+2]))[1:-1]
        y = np.linspace(0, 1, np.max([3, n_vec[1]+2]))[1:-1]
        z = np.linspace(0, 1, np.max([3, n_vec[2]+2]))[1:-1]
        # transform to box
        points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        return self._transform_unit_cube_to_box(points)

    def _random_sampling_boundary(self, n):
        n_vec = self._divide_points_depending_on_area(n)
        points = np.zeros((0, 3))
        # use permutation to change the roll of x,y and z:
        # permut[0]: z = 0 or 1, permut[1]: y = 0 or 1, ...
        permut = [[0, 1, 2], [0, 2, 1], [2, 0, 1]]
        for i in range(3):
            new_points = self._random_points_on_boundary(n_vec[i], permut[i])
            points = np.append(points, new_points, axis=0)
        if len(points) < n:
            max_side = np.argmax(self.side_areas)
            new_points = self._random_points_on_boundary(n-len(points),
                                                         permut[max_side])
            points = np.append(points, new_points, axis=0)                                            
        return points.astype(np.float32)

    def _random_points_on_boundary(self, n, permut):
        # first on unit cube
        x = np.random.uniform(0, 1, (n,1))
        y = np.random.uniform(0, 1, (n,1))
        z = np.random.randint(0, 2, (n,1)) # here just 0 or 1
        new_points = np.column_stack((x, y, z))[:, permut]
        # then transform to box
        new_points = self._transform_unit_cube_to_box(new_points)
        return new_points

    def _divide_points_depending_on_area(self, n):
        # divide number of sampling points, depending on the side area
        area_sum = np.sum(self.side_areas)
        n_vec = np.zeros(3, dtype=int)
        for i in range(3):
            n_vec[i] = int(n*self.side_areas[i]/area_sum)   
        return n_vec

    def _grid_sampling_boundary(self, n):
        # Take only n/2, since we have two grids on opposing sides
        n_vec = self._divide_points_depending_on_area(int(n/2))
        points = np.zeros((0, 3))
        # use permutation to change the roll of x,y and z:
        # permut[0]: z = 0 or 1, permut[1]: y = 0 or 1, ...
        permut = [[0, 1, 2], [0, 2, 1], [2, 0, 1]]
        root_of_n = np.sqrt(n_vec).astype(int)
        # to get not multiple points at the edges, delete the first and last 
        # points. Depending on step i
        num_of_lins = [[range(root_of_n[0]), range(root_of_n[0])], 
                       [range(root_of_n[1]), range(1, root_of_n[1]+1)],
                       [range(1, root_of_n[2]+1), range(1, root_of_n[2]+1)]]
        # if we delete points, take some extra steps to have the right number 
        # of samples
        add_n = [[0, 0], [0, 2], [2, 2]]
        for i in range(3):
            # first on unit cube
            x = np.linspace(0, 1, root_of_n[i]+add_n[i][0])[num_of_lins[i][0]]
            y = np.linspace(0, 1, root_of_n[i]+add_n[i][1])[num_of_lins[i][1]]
            z = [0, 1]
            new_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            points = np.append(points, new_points[:, permut[i]], axis=0)
        # then transform to box
        points = self._transform_unit_cube_to_box(points)
        # sample on the biggest erea some random points, if there are some 
        # missing: 
        if len(points) < n:
            max_side = np.argmax(self.side_areas)
            new_points = self._random_points_on_boundary(n-len(points),
                                                         permut[max_side])
            points = np.append(points, new_points, axis=0)                                            
        return points.astype(np.float32)

    def serialize(self):
        """to show data/information in tensorboard
        """
        dct = super().serialize()
        dct['name'] = 'Box'
        dct['origin'] = [int(a) for a in list(self.corner_o)]
        dct['x-corner'] = [int(a) for a in list(self.corner_x)]
        dct['y-corner'] = [int(a) for a in list(self.corner_y)]
        dct['z-corner'] = [int(a) for a in list(self.corner_z)]
        return dct

    def _compute_bounds(self):
        """computes bounds of the domain

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        # all corners of the box
        corners = np.array([self.corner_o, self.corner_x, self.corner_y,
                            self.corner_y-self.corner_o+self.corner_x, 
                            self.corner_z,
                            self.corner_y-self.corner_o+self.corner_z, 
                            self.corner_x-self.corner_o+self.corner_z, 
                            self.corner_y-self.corner_o+self.corner_x
                            + self.corner_z-self.corner_o])
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        min_z = np.min(corners[:, 2])
        max_z = np.max(corners[:, 2])
        return [min_x, max_x, min_y, max_y, min_z, max_z]

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector
            has to be computed, e.g. [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        index = np.zeros((len(points), 6))
        normals = np.zeros((len(points), 3))
        points = self._transform_to_unit_cube(points) 
        k = 0
        for i in range(0,6,2):
            # check if points on the x,y or z boundary
            index[:, i] = np.isclose(points[:, k], 0, atol=self.tol)
            index[:, i+1] = np.isclose(points[:, k], 1, atol=self.tol)
            k = k + 1
            # add the normal vectors:
            normals += (self.normals[i+1] * index[:, i:i+1]
                        + self.normals[i] * index[:, i+1:i+2])
        # check if some corner or edge points exist. Then more then one
        # index is not zero
        sum_index = np.sum(index, axis=1)
        if np.any(sum_index == 0):
            print('Warning: some points are not at the boundary!')    
        index_2 = np.where(sum_index == 2)[0]
        index_3 = np.where(sum_index == 3)[0]
        normals[index_2] /= np.sqrt(2)
        normals[index_3] /= np.sqrt(3)
        return normals.astype(np.float32)


class Sphere(Domain):
    """Class for arbitrary spheres.

    Parameters
    ----------
    center : array_like
        The center of the sphere, e.g. center = [5,0,1].
    radius : number
        The radius of the sphere.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    """
    def __init__(self, center, radius, tol=1e-06):
        super().__init__(dim=3, volume=4/3*np.pi*radius**2,
                         surface=4*np.pi*radius**2, tol=tol)
        self.center = np.asarray(center)
        self.radius = radius

    def is_inside(self, points):
        """Checks if the given points are inside the open sphere.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was inside, or false if not.
        """
        points = np.subtract(points, self.center)
        inside = np.linalg.norm(points, axis=1) < (self.radius + self.tol)
        return inside.reshape(-1, 1)

    def is_on_boundary(self, points):
        """Checks if the given points are on the boundary of the sphere.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was on the boundary, or false if not.
        """
        points = np.subtract(points, self.center)
        on_bound = np.isclose(np.linalg.norm(points, axis=1),
                              self.radius, atol=self.tol)        
        return on_bound.reshape(-1, 1)

    def _compute_bounds(self):
        """computes bounds of the sphere.

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        low_b = np.subtract(self.center, self.radius)
        up_b = np.add(self.center, self.radius)
        return [low_b[0], up_b[0], low_b[1], up_b[1], low_b[2], up_b[2]]

    def _random_sampling_boundary(self, n):
        # use polar-coords.
        phi = np.random.uniform(0, 2*np.pi, n).reshape(-1, 1)
        # choose theta so that the points are uniform
        theta = (np.arccos(2*np.random.uniform(0, 1, n)-1)-np.pi/2.0).reshape(-1, 1)
        x = self.radius*np.cos(phi)*np.cos(theta)
        y = self.radius*np.sin(phi)*np.cos(theta)
        z = self.radius*np.sin(theta)
        points = np.concatenate((x,y,z), axis=1)
        return np.add(points, self.center).astype(np.float32)

    def _grid_sampling_boundary(self, n):
        # From:
        # https://stackoverflow.com/questions/9600801/
        # evenly-distributing-n-points-on-a-sphere
        points = []
        # Use Fibonacci-Sphere
        phi = np.pi * (3.0 - np.sqrt(5.0)) # golden angle in radians
        for i in range(n):
            y = 1 - (i/float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        # Translate to center and scale
        points = np.array(points).reshape(-1, 3)
        points = np.add(self.radius*points, self.center)
        return points.astype(np.float32)

    def _random_sampling_inside(self, n):
        phi = np.random.uniform(0, 2*np.pi, n).reshape(-1, 1)
        theta = (np.arccos(2*np.random.uniform(0, 1, n)-1)-np.pi/2.0).reshape(-1, 1)
        r = np.random.uniform(0, 1, n)
        r = self.radius*np.cbrt(r).reshape(-1, 1) # take cubic-root, to stay uniform
        x = r*np.cos(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.cos(theta)
        z = r*np.sin(theta)
        points = np.concatenate((x,y,z), axis=1)
        return np.add(points, self.center).astype(np.float32)

    def _grid_sampling_inside(self, n):
        points = self._point_grid_in_sphere(n)
        # if not enough points add random ones
        points = super()._check_inside_grid_enough_points(n, points)
        # to many -> delete some
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def _point_grid_in_sphere(self, n):
        # sample in bounding box and remove points outside
        scaled_n = int(np.cbrt(n*6/np.pi))
        axis = np.linspace(-self.radius, self.radius, scaled_n+2)[1:-1]
        points = np.array(np.meshgrid(axis, axis, axis)).T.reshape(-1, 3)
        points = np.add(points, self.center)
        inside = np.nonzero(self.is_inside(points))[0]
        return points[inside]

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector
            has to be computed, e.g. [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        if not all(self.is_on_boundary(points)):
            print('Warning: some points are not at the boundary!')
        normal_vectors = np.subtract(points, self.center) / self.radius
        return normal_vectors.astype(np.float32)

    def serialize(self):
        """to show data/information in tensorboard
        """
        dct = super().serialize()
        dct['name'] = 'Sphere'
        dct['center'] = [int(a) for a in list(self.center)]
        dct['radius'] = self.radius
        return dct

    def grid_for_plots(self, n):
        """Creates a grid of points for plotting. (grid at boundary + inside)
        """
        #points_inside = self._point_grid_in_circle(int(np.ceil(3*n/4)))
        # add some points at the boundary to better show the form of the circle
        #points_boundary = self._grid_sampling_boundary(int(n/4))
        #points = np.append(points_inside, points_boundary, axis=0)
        #return points.astype(np.float32)


class Cylinder(Domain):
    """Class for arbitrary cylinders.

    Parameters
    ----------
    center : array_like
        The center of the cylinder, e.g. center = [5,0,1].
    radius : number
        The radius of the cylinder.
    height : number
        The total height of the cylinder.
    orientation : array_like
        The orientation of the cylinder. A vector that is orthognal to the circle
        areas of the cylinder.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary.
    """
    def __init__(self, center, radius, height, orientation, tol=1e-06):
        super().__init__(dim=3, volume=np.pi*radius**2*height,
                         surface=2*np.pi*(radius**2+radius*height),
                         tol=tol)
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.orientation = np.array(orientation)
        norm_ori = np.linalg.norm(orientation)
        if not norm_ori == 1:
            self.orientation = np.divide(self.orientation, norm_ori)
        self.rotation_matrix = self._create_rotation_matrix()

    def _create_rotation_matrix(self):
        # create a matrix to rotate the cylinder, to be parallel to the z-axis
        # From:
        # https://math.stackexchange.com/questions
        # /180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        c = -self.orientation[2] # cosine of angle between orientation and (0,0,-1)
        v = [-self.orientation[1], self.orientation[0], 0] # cross pord. 
        if c == -1: # orientation is (0,0,1) -> do nothing
            return np.eye(3)
        else: 
            I = np.eye(3)
            M = np.array([[0, 0, v[1]], [0, 0, -v[0]], [-v[1], v[0], 0]])
            R = I + M + 1/(1+c)*np.linalg.matrix_power(M, 2)
            return R

    def _transform_points_to_origin_and_rotate(self, points):
        points = np.subtract(points, self.center)
        points = np.array([np.matmul(self.rotation_matrix, i) for i in points])
        return points

    def _transform_points_into_cylinder(self, points):
        inverse_rot = self.rotation_matrix.T
        points = np.array([np.matmul(inverse_rot, i) for i in points])
        points = np.add(points, self.center)
        return points

    def is_inside(self, points):
        """Checks if the given points are inside the cylinder.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was inside, or false if not.
        """
        points = self._transform_points_to_origin_and_rotate(points)
        norm_points = np.linalg.norm(points[:, 0:2], axis=1)
        return self._inside_cylinder(points, norm_points).reshape(-1, 1)

    def _inside_cylinder(self, points, norm_points):
        norm_inside = norm_points <= self.radius + self.tol
        height_inside = np.abs(points[:, 2]) <= self.height/2 + self.tol
        return np.logical_and(norm_inside, height_inside)

    def is_on_boundary(self, points):
        """Checks if the given points are on the boundary of the sphere.

        Parameters
        ----------
        points : list of lists
            A list containing all points that have to be checked.
            The list has to be of the form [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains either true,
            if the points was on the boundary, or false if not.
        """
        points = self._transform_points_to_origin_and_rotate(points)
        norm_points = np.linalg.norm(points[:, 0:2], axis=1)
        norm_on_bound = np.isclose(norm_points, self.radius, atol=self.tol)
        height_on_bound = np.isclose(np.abs(points[:, 2]),
                                     self.height/2, atol=self.tol)
        on_bound = np.logical_or(norm_on_bound, height_on_bound)
        inside = self._inside_cylinder(points, norm_points)
        return np.logical_and(inside, on_bound).reshape(-1, 1)

    def _random_sampling_inside(self, n):
        circle = Circle([0, 0], radius=self.radius)
        points = circle._random_sampling_inside(n)
        z = np.random.uniform(-self.height/2, self.height/2, (n,1))
        points = np.column_stack((points, z))
        return self._transform_points_into_cylinder(points).astype(np.float32)

    def _grid_sampling_inside(self, n):
        points = self._grid_in_cylinder(n)
        points = super()._check_inside_grid_enough_points(n, points)
        return points

    def _grid_in_cylinder(self, n):
        # use a bounding box to sample a grid:
        r = self.radius
        h = self.height/2
        box = Box(corner_o=[-r, -r, -h], corner_x=[r, -r, -h], 
                  corner_y=[-r, r, -h], corner_z=[-r, -r, h])
        points = box._grid_in_box(int(n*4/np.pi))
        # check what points have the correct position
        norm = np.linalg.norm(points[:, :2], axis=1)
        index = np.where(norm <= r + self.tol)[0]
        # tranfsform these points into the clyinder 
        points = self._transform_points_into_cylinder(points[index])
        return points.astype(np.float32)

    def _random_sampling_boundary(self, n):
        # first divide n for points on the cricle areas and the outer 
        # surface.
        n_circle = int(n*self.radius/(self.height+self.radius))
        circle = Circle([0, 0], radius=self.radius)
        # first in the circle areas:
        xy_points = circle._random_sampling_inside(n_circle)
        z = np.random.randint(0, 2, (n_circle, 1)) 
        points = np.column_stack((xy_points,
                                  np.subtract(self.height*z, self.height/2)))
        # now points on the lateral surface
        n = n - n_circle
        xy_points = circle._random_sampling_boundary(n)
        z = np.random.uniform(-self.height/2, self.height/2, (n, 1))
        new_points = np.column_stack((xy_points, z))
        # put points together and transform into right form
        points = np.concatenate((points, new_points))
        points = self._transform_points_into_cylinder(points)
        return points.astype(np.float32)

    def _grid_sampling_boundary(self, n):
        # first divide n for points on the cricle areas and the outer 
        # surface.
        n_circle = int(n*self.radius/(self.height+self.radius))
        circle = Circle([0, 0], radius=self.radius)
        # first in the circle areas:
        n_divide = int(np.ceil(n_circle/2)) # we need a grid at the top and bottom
        xy_points = circle._point_grid_in_circle(n_divide)
        points_1 = np.column_stack((xy_points, len(xy_points)*[-self.height/2]))
        points_2 = np.column_stack((xy_points, len(xy_points)*[self.height/2]))
        points = np.concatenate((points_1, points_2))
        # now points on the lateral surface:
        n_lateral = int(np.sqrt(n - n_circle))
        xy_points = circle._grid_sampling_boundary(n_lateral)
        z = np.linspace(-self.height/2, self.height/2, n_lateral)
        # combine the xy_points and z through a meshgrid
        index = range(0, n_lateral)
        index = np.array(np.meshgrid(index, index)).T.reshape(-1, 2)
        new_points = np.column_stack((xy_points[index[:, 0]], z[index[:, 1]]))
        # put points together and transform into right form
        points = np.concatenate((points, new_points))
        points = self._transform_points_into_cylinder(points)
        points = super()._check_boundary_grid_enough_points(n, points)
        points = super()._cut_points(n, points)
        return points.astype(np.float32)

    def boundary_normal(self, points):
        '''Computes the boundary normal.

        Parameters
        ----------
        points : list of lists
            A list containing all points where the normal vector
            has to be computed, e.g. [[x1,y1,z1],[x2,y2,z2],...].

        Returns
        ----------
        np.array
            Every entry of the output contains the normal vector at the point,
            specified in the input array.
        '''
        normals = np.zeros((len(points), 3))
        index = np.zeros((len(points), 3))
        points = self._transform_points_to_origin_and_rotate(points)
        # first check the top and bottom circle:
        index[:, 0] = np.isclose(points[:, 2], self.height/2, atol=self.tol) 
        index[:, 1] = np.isclose(points[:, 2], -self.height/2, atol=self.tol)
        normals += [0, 0, 1] * (index[:, :1] - index[:, 1:2])
        # check if points lay on the lateral surface:
        norm = np.linalg.norm(points[:, :2], axis=1)
        index[:, 2] = np.isclose(norm, self.radius, atol=self.tol)
        i = np.where(index[:, 2])[0]
        if not len(i) == 0:
            normals[i] += np.column_stack((points[i, :2]/self.radius,
                                           len(i)*[0]))
        # check if some edge points exist. Then more then one
        # index is not zero
        sum_index = np.sum(index, axis=1)
        if np.any(sum_index == 0):
            print('Warning: some points are not at the boundary!')    
        i = np.where(sum_index == 2)[0]
        normals[i] /= np.sqrt(2)
        # rotate the normal vectors:
        normals = np.array([np.matmul(self.rotation_matrix.T, n) for n in normals])
        return normals.astype(np.float32)

    def _compute_bounds(self):
        """computes bounds of the cylinder.

        Returns
        -------
        np.array:
            The bounds in the form: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        # first take size in the direction of the orientation
        A = self.center + self.height/2 * self.orientation
        B = self.center - self.height/2 * self.orientation
        max_b = np.max((A, B), axis=0)
        min_b = np.min((A, B), axis=0)
        # second condsider the radius of the cylinder
        dif = A - B
        norm_dif = np.linalg.norm(dif)
        x = np.sqrt((dif[1]**2+dif[2]**2)/norm_dif**2)
        y = np.sqrt((dif[0]**2+dif[2]**2)/norm_dif**2)
        z = np.sqrt((dif[1]**2+dif[0]**2)/norm_dif**2)
        C = [self.radius*x, self.radius*y, self.radius*z]
        min_b -= C
        max_b += C
        return [min_b[0], max_b[0], min_b[1], max_b[1], min_b[2], max_b[2]]

    def serialize(self):
        """to show data/information in tensorboard
        """
        dct = super().serialize()
        dct['name'] = 'Cylinder'
        dct['center'] = [int(a) for a in list(self.center)]
        dct['radius'] = self.radius
        dct['height'] = self.height
        return dct


class Polygon3D(Domain):
    '''Class for polygons in 3D.

    Parameters
    ----------
    vertices : list of lists, optional 
        The vertices of the polygon.
    faces : list of lists, optional 
        A list that contains which vetrices have to be connected to create the faces
        of the polygon. If for example the vertices 1, 2 and 3 have should be 
        connected do: faces = [[1,2,3]]
    file_name : str or file-like object, optional
        A data source to load a existing polygon/mesh.
    file_type : str, optional
        The file type, e.g. 'stl'. See trimesh.available_formats() for all supported
        file types.
    tol : number, optional
        The error tolerance for checking if points are inside or at the boundary
    '''
    def __init__(self, vertices=None, faces=None, file_name=None, file_type='stl',
                 tol=1e-06):
        if vertices is not None and faces is not None:
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh.fix_normals()
        elif file_name is not None:
            self.mesh = trimesh.load_mesh(file_name, file_type=file_type)
        else:
            raise ValueError('Needs either vertices and faces to create a new' \
                             'polygon, or a file to load a existing one.')
        super().__init__(dim=3, volume=self.mesh.volume, 
                         surface=sum(self.mesh.area_faces), tol=tol)
        # Trimesh gives a warning when not enough points are sampled. We already
        # take care of this problem. So set the logging only to errors.
        logging.getLogger("trimesh").setLevel(logging.ERROR)

    def export_file(self, name_of_file):
        '''Exports the mesh to a file.

        Parameters
        ----------
        name_of_file : str
            The name of the file.
        '''
        self.mesh.export(name_of_file)

    def project_on_plane(self, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
        '''Projects the polygon on a plane. 

        Parameters
        ----------
        plane_origin : array_like, optional
            The origin of the projection plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 

        Returns
        ----------
        Polygon2D
            The polygon that is the outline of the projected original mesh on 
            the plane.
        '''
        norm = np.linalg.norm(plane_normal)
        if not np.isclose(norm, 1):
            plane_normal /= norm
        polygon = trimesh.path.polygons.projected(self.mesh, origin=plane_origin,
                                                  normal=plane_normal)
        polygon = polygon.simplify(self.tol)
        return Polygon2D(shapely_polygon=polygon, tol=self.tol)

    def slice_with_plane(self, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
        '''Slices the polygon with a plane.

        Parameters
        ----------
        plane_origin : array_like, optional
            The origin of the plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 

        Returns
        ----------
        Polygon2D
            The polygon that is the outline of the projected original mesh on 
            the plane.
        '''
        norm = np.linalg.norm(plane_normal)
        if not np.isclose(norm, 1):
            plane_normal /= norm
        rotaion_matrix = self._create_rotation_matrix_to_plane(plane_normal)
        slice = self.mesh.section(plane_origin=plane_origin,
                                  plane_normal=plane_normal)
        if slice is None:
            raise ValueError('slice of mesh and plane is empty!')
        slice_2D = slice.to_planar(to_2D=rotaion_matrix, check=False)[0]
        polygon = slice_2D.polygons_full[0]
        polygon = polygon.simplify(self.tol)
        return Polygon2D(shapely_polygon=polygon, tol=self.tol)

    def _create_rotation_matrix_to_plane(self, plane_normal):
        u = [plane_normal[1], -plane_normal[0], 0]
        cos = plane_normal[2]
        sin = np.sqrt(plane_normal[0]**2 + plane_normal[1]**2)
        matrix = [[cos+u[0]**2*(1-cos), u[0]*u[1]*(1-cos),    -u[1]*sin, 0], 
                  [u[0]*u[1]*(1-cos),   cos+u[1]**2*(1-cos),  u[0]*sin,  0], 
                  [-u[1]*sin,           u[0]*sin,             -cos,      0],
                  [0,                   0,                    0,         1]]
        return matrix

    def is_inside(self, points):
        '''Checks if the given points are inside the mesh.

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
        return self.mesh.contains(points).reshape(-1,1)

    def is_on_boundary(self, points):
        '''Checks if the given points are on the surface of the mesh.

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
        distance = trimesh.proximity.signed_distance(self.mesh, points)
        abs_dist = np.absolute(distance)
        on_bound = (abs_dist <= self.tol)
        return on_bound.reshape(-1,1)

    def _random_sampling_inside(self, n):
        points = np.empty((0,self.dim))
        missing = n
        while len(points) < n:
            new_points = trimesh.sample.volume_mesh(self.mesh, missing)
            points = np.append(points, new_points, axis=0)
            missing -= len(new_points)
        return points.astype(np.float32)

    def _grid_sampling_inside(self, n):
        raise NotImplementedError #Needs 3D Box class

    def _random_sampling_boundary(self, n):
        return trimesh.sample.sample_surface(self.mesh, n)[0].astype(np.float32)

    def _grid_sampling_boundary(self, n):
        points = trimesh.sample.sample_surface_even(self.mesh, n)[0]
        points = super()._check_boundary_grid_enough_points(n, points)
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
        index = self.mesh.nearest.on_surface(points)[2]
        mesh_normals = self.mesh.face_normals
        normals = np.zeros((len(points), self.dim))
        for i in range(len(points)):
            normals[i, :] = mesh_normals[index[i]]
        return normals.astype(np.float32)