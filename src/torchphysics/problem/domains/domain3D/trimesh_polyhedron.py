import numpy as np
import torch
import trimesh
import logging

from ..domain import Domain, BoundaryDomain
from ..domain2D.shapely_polygon import ShapelyPolygon
from .sphere import Sphere
from ...spaces import Points


class TrimeshPolyhedron(Domain):
    '''Class for polygons in 3D. Uses the trimesh-package.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    vertices : list of lists, optional 
        The vertices of the polygon.
    faces : list of lists, optional 
        A list that contains which vetrices have to be connected to create the faces
        of the polygon. If for example the vertices 1, 2 and 3 have should be 
        connected do: faces = [[1, 2, 3]]
    file_name : str or file-like object, optional
        A data source to load a existing polygon/mesh.
    file_type : str, optional
        The file type, e.g. 'stl'. See trimesh.available_formats() for all supported
        file types.
    tol : number, optional
        The error tolerance for checking if points at the boundary. And used for 
        projections and slicing the mesh.

    Note
    ----
    This class can not be dependent on other variables.
    '''
    def __init__(self, space, vertices=None, faces=None,
                 file_name=None, file_type='stl', tol=1.e-06):
        assert space.dim == 3
        if vertices is not None and faces is not None:
            if callable(vertices) or callable(faces):
                raise TypeError("""Trimesh can not use functions as vertices/faces.""")
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        elif file_name is not None:
            self.mesh = trimesh.load_mesh(file_name, file_type=file_type)
        else:
            raise ValueError('Needs either vertices and faces to create a new' \
                             'polygon, or a file to load a existing one.')
        self.mesh.fix_normals()
        super().__init__(space, dim=3)
        self.necessary_variables = {}
        self.tol = tol
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

    def project_on_plane(self, new_space, plane_origin=[0, 0, 0], 
                         plane_normal=[0, 0, 1]):
        '''Projects the polygon on a plane. 

        Parameters
        ----------
        new_space : Space
            The space in which the projected object should lay.
        plane_origin : array_like, optional
            The origin of the projection plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 
        
        Returns
        ----------
        ShapelyPolygon
            The polygon that is the outline of the projected original mesh on 
            the plane.
        '''
        norm = np.linalg.norm(plane_normal)
        if not np.isclose(norm, 1):
            plane_normal /= norm
        polygon = trimesh.path.polygons.projected(self.mesh, origin=plane_origin,
                                                  normal=plane_normal)
        polygon = polygon.simplify(self.tol)
        return ShapelyPolygon(space=new_space, shapely_polygon=polygon)

    def slice_with_plane(self, new_space, plane_origin=[0, 0, 0],
                         plane_normal=[0, 0, 1]):
        '''Slices the polygon with a plane.

        Parameters
        ----------
        new_space : Space
            The space in which the projected object should lay.
        plane_origin : array_like, optional
            The origin of the plane.
        plane_normal : array_like, optional
            The normal vector of the projection plane. It is enough if it points in the
            direction of normal vector, it does not norm = 1. 
        
        Returns
        ----------
        ShapelyPolygon
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
            raise RuntimeError('slice of mesh and plane is empty!')
        slice_2D = slice.to_planar(to_2D=rotaion_matrix, check=False)[0]
        polygon = slice_2D.polygons_full[0]
        polygon = polygon.simplify(self.tol)
        return ShapelyPolygon(space=new_space, shapely_polygon=polygon)

    def _create_rotation_matrix_to_plane(self, plane_normal):
        u = [plane_normal[1], -plane_normal[0], 0]
        cos = plane_normal[2]
        sin = np.sqrt(plane_normal[0]**2 + plane_normal[1]**2)
        matrix = [[cos+u[0]**2*(1-cos), u[0]*u[1]*(1-cos),    -u[1]*sin, 0], 
                  [u[0]*u[1]*(1-cos),   cos+u[1]**2*(1-cos),  u[0]*sin,  0], 
                  [-u[1]*sin,           u[0]*sin,             -cos,      0],
                  [0,                   0,                    0,         1]]
        return matrix

    def __call__(self, **data):
        return self

    def bounding_box(self, params=Points.empty(), device='cpu'):
        bound_corners = self.mesh.bounds
        return torch.tensor(bound_corners.T.flatten(), device=device, 
                            dtype=torch.float32)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        volume = self.mesh.volume
        return torch.tensor(volume, device=device).reshape(-1, 1)

    def _contains(self, points, params=Points.empty()):
        if isinstance(points, Points):
            points = points.as_tensor
        inside = self.mesh.contains(points).reshape(-1,1)
        return torch.tensor(inside)

    def _compute_number_of_points(self, n, d, params):
        if d:
            n = self.compute_n_from_density(d, params)
        num_of_params = self.len_of_params(params)
        n *= num_of_params
        return n

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(),
                              device='cpu'):
        n = self._compute_number_of_points(n, d, params)
        points = torch.empty((0, self.dim), dtype=torch.float32, device=device)
        computed_points = 0
        while computed_points < n:
            new_points = trimesh.sample.volume_mesh(self.mesh, n-computed_points)
            points = torch.cat((points, torch.tensor(new_points, device=device, 
                                                     dtype=torch.float32)),dim=0)
            computed_points += len(new_points)
        return Points(points, self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        n = self._compute_number_of_points(n, d, params)
        bounds = self.bounding_box(params, device=device)
        points = self._point_grid_in_bounding_box(n, bounds, device)
        points_inside = self._get_points_inside(points)
        final_points = Sphere._append_random(self, points_inside, n, params, device)
        return Points(final_points, self.space)

    def _point_grid_in_bounding_box(self, n, bounds, device):
        b_box_volume = self._get_bounding_box_volume(bounds)
        volume = self._get_volume(device=device).item()
        scaled_n = int(np.ceil(np.cbrt(n*b_box_volume/volume)))
        x_axis = torch.linspace(bounds[0], bounds[1], scaled_n, device=device)
        y_axis = torch.linspace(bounds[2], bounds[3], scaled_n, device=device)
        z_axis = torch.linspace(bounds[4], bounds[5], scaled_n, device=device)
        points = torch.stack(torch.meshgrid(x_axis, y_axis, z_axis)).mT
        return points.reshape(-1, 3)

    def _get_bounding_box_volume(self, bounds):
        b_box_volume = 1
        for i in range(self.dim):
            b_box_volume *= bounds[2*i+1] - bounds[2*i]
        return b_box_volume

    def _get_points_inside(self, points):
        inside = self._contains(points)
        index = torch.where(inside)[0]
        return points[index]

    @property
    def boundary(self):
        return TrimeshBoundary(self)


class TrimeshBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, TrimeshPolyhedron)
        super().__init__(domain)

    def _contains(self, points, params=Points.empty()):
        points = points.as_tensor
        distance = trimesh.proximity.signed_distance(self.domain.mesh, points)
        abs_dist = torch.absolute(torch.tensor(distance))
        on_bound = (abs_dist <= self.domain.tol)
        return on_bound.reshape(-1,1)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        area = sum(self.domain.mesh.area_faces)
        return torch.tensor(area, device=device).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        n = self.domain._compute_number_of_points(n, d, params)
        points = trimesh.sample.sample_surface(self.domain.mesh, n)[0]
        tensor_points = torch.tensor(points, device=device, dtype=torch.float32)
        return Points(tensor_points, self.space)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        n = self.domain._compute_number_of_points(n, d, params)
        points = trimesh.sample.sample_surface_even(self.domain.mesh, n)[0]
        points = torch.tensor(points, device=device, dtype=torch.float32)
        points = Sphere._append_random(self, points, n, params, device)
        return Points(points, self.space)

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        points = points.as_tensor.detach().cpu()
        index = self.domain.mesh.nearest.on_surface(points)[2]
        mesh_normals = torch.tensor(self.domain.mesh.face_normals, device=device)
        normals = torch.zeros((len(points), 3), device=device)
        for i in range(len(points)):
            normals[i, :] = mesh_normals[index[i]]
        return normals