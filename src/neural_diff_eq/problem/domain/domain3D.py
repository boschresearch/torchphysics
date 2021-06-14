import numpy as np
import trimesh
import logging

from .domain import Domain
from .domain2D import Polygon2D


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
        points = trimesh.sample.sample_surface_even(self.mesh, n)[0].astype(np.float32)
        if len(points) < n:
            points = np.concatenate((points,
                                     self._random_sampling_boundary(n-len(points))))
        return points

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