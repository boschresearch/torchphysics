import torch
import shapely.geometry as s_geo
import shapely.ops as s_ops

from ..domain import Domain, BoundaryDomain
from .parallelogram import Parallelogram
from ...spaces import Points

class ShapelyPolygon(Domain):
    """Class for polygons. Uses the shapely-package.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    vertices : list of lists, optional 
        The corners/vertices of the polygon. Can be eihter in clockwise or counter-
        clockwise order. 
    shapely_polygon : shapely.geometry.Polygon, optional
        Instead of defining the corner points, it is also possible to give a already
        existing shapely.Polygon object. 

    Note
    ----
    This class can not be dependent on other variables.
    """
    def __init__(self, space, vertices=None, shapely_polygon=None):
        assert space.dim == 2
        super().__init__(space, dim=2)
        self.necessary_variables = self.set_necessary_variables()
        if isinstance(shapely_polygon, s_geo.Polygon):
            self.polygon = shapely_polygon
        elif vertices:
            if callable(vertices):
                TypeError("""Shapely-Polygons can not use functions as vertices.""")
            self.polygon= s_geo.Polygon(vertices)
        else:
            raise ValueError("""Needs either vertices or a shapely polygon as input""")
        self.polygon = s_geo.polygon.orient(self.polygon)

    def __call__(self, **data):
        return self

    def _contains(self, points, params=Points.empty()):
        if isinstance(points, Points):
            points = points.as_tensor
        inside = torch.zeros(len(points), 1)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            inside[i] = self.polygon.contains(point)
        return inside

    def bounding_box(self, device='cpu'):
        bounds = torch.tensor(self.polygon.bounds, device=device)
        bounds[[1,2]] = bounds[[2,1]]
        return bounds

    def _get_volume(self, params=Points.empty(), device='cpu'):
        volume = self.polygon.area
        return torch.tensor(volume, device=device).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        n = self._compute_number_of_points(n, d, params)
        points = torch.empty((0, self.dim), device=device)
        big_t, biggest_area = None, 0
        # instead of using a bounding box it is more efficient to triangulate
        # the polygon and sample in each triangle.
        for t in s_ops.triangulate(self.polygon):
            scaled_n = int(t.area/self.polygon.area * n)
            new_points = self._sample_in_triangulation(t, scaled_n, device)
            if new_points is not None:
                points = torch.cat((points, new_points), dim=0)
                # remember the biggest triangle that was inside, if later
                # some additional points need to be added
                if t.within(self.polygon) and t.area > biggest_area:
                    big_t = [t][0]
                    biggest_area = t.area
            if len(points) == n:
                break
        points = self._check_enough_points_sampled(n, points, big_t, device)
        return Points(points, self.space)

    def _sample_in_triangulation(self, t, n, device):
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        corners = torch.tensor([[x0, y0], [x1, y1], [x2, y2]], device=device)
        if n > 0:
            new_points = self._random_points_in_triangle(n, corners, device)
            # when the polygon has holes or is non convex, it can happen
            # that the triangle is not completly in the polygon
            if not t.within(self.polygon):
                inside = self._contains(new_points)
                index = torch.where(inside)[0]
                new_points = new_points[index]
            return new_points
        return None

    def _random_points_in_triangle(self, n, corners, device):
        bary_coords = torch.rand((n, 2), device=device)
        # if a barycentric coordinates is bigger then 1, mirror them at the
        # point (0.5, 0.5). Stays uniform.
        index = torch.where(bary_coords.sum(axis=1) > 1)[0]
        bary_coords[index] = torch.subtract(torch.tensor([[1.0, 1.0]], device=device),
                                            bary_coords[index])
        axis_1 = torch.multiply(corners[1]-corners[0], bary_coords[:, :1])
        axis_2 = torch.multiply(corners[2]-corners[0], bary_coords[:, 1:])
        return torch.add(torch.add(corners[0], axis_1), axis_2)

    def _check_enough_points_sampled(self, n, points, big_t, device):
        # if not enough points are sampled, create some new points in the biggest
        # triangle
        while len(points) < n:
            new_points = self._sample_in_triangulation(big_t, n-len(points), device)
            points = torch.cat((points, new_points), dim=0)
        return points

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        n = self._compute_number_of_points(n, d, params)
        points = self._create_points_in_bounding_box(n, device)
        points = self._delete_outside(points)
        if not d:
            # if a number of points if specified we have to make sure
            # to sample the right amount of points
            points = self._grid_enough_points(n, points, device)
        return Points(points, self.space)

    def _create_points_in_bounding_box(self, n, device):
        bounds = self.bounding_box(device=device)
        origin = bounds[[0,2]]
        dir_1 = torch.tensor([[bounds[1]-bounds[0], 0]], device=device)
        dir_2 = torch.tensor([[0, bounds[3]-bounds[2]]], device=device)
        b_box_volume = (bounds[1]-bounds[0])*(bounds[3]-bounds[2])
        scaled_n = int(n * b_box_volume/self.polygon.area)
        b_box_grid = Parallelogram._compute_barycentric_grid(self, scaled_n,
                                                             dir_1, dir_2, device)
        points_in_dir_1 = b_box_grid[:, :1] * dir_1
        points_in_dir_2 = b_box_grid[:, 1:] * dir_2
        points = points_in_dir_1 + points_in_dir_2 + origin
        return points

    def _delete_outside(self, points):
        inside = self._contains(points)
        index = torch.where(inside)[0]
        return points[index]

    def _grid_enough_points(self, n, bary_coords, device): 
        # if not enough points, add some random ones.
        points = bary_coords
        if len(bary_coords) < n:
            random_points = self.sample_random_uniform(n=(n - len(bary_coords)),
                                                       device=device)
            points = torch.cat((bary_coords, random_points.as_tensor),
                                dim=0)
        return points

    def _compute_number_of_points(self, n, d, params):
        if d:
            n = self.compute_n_from_density(d, params)
        num_of_params = self.len_of_params(params)
        n *= num_of_params
        return n

    def outline(self, device='cpu'):
        """Creates a outline of the domain.

        Returns
        -------
        list of list
            The vertices of the domain. Inner vertices are appended in there
            own list.
        """
        cords = [torch.tensor(self.polygon.exterior.coords, device=device)]
        for i in self.polygon.interiors:
            cords.append(torch.tensor(i.coords, device=device))
        return cords

    @property
    def boundary(self):
        return ShapelyBoundary(self)


class ShapelyBoundary(BoundaryDomain):

    def __init__(self, domain):
        assert isinstance(domain, ShapelyPolygon)
        super().__init__(domain)
        outline = self.domain.outline()
        self.normal_list = self._compute_normals(outline)
        self.tol = 1.e-06

    def __call__(self, **data):
        return self

    def _contains(self, points, params=Points.empty()):
        points = points.as_tensor
        on_bound = torch.empty(len(points), dtype=bool, device=points.device)
        for i in range(len(points)):
            point = s_geo.Point(points[i])
            distance = self.domain.polygon.boundary.distance(point)
            on_bound[i] = (abs(distance) <= self.tol)
        return on_bound.reshape(-1, 1)

    def _get_volume(self, params=Points.empty(), device='cpu'):
        volume = self.domain.polygon.boundary.length 
        return torch.tensor(volume, device=device).reshape(-1, 1)

    def sample_random_uniform(self, n=None, d=None, params=Points.empty(), 
                              device='cpu'):
        n = self.domain._compute_number_of_points(n, d, params)
        line_points = torch.rand(n, device=device) * self.domain.polygon.boundary.length 
        return self._transform_points_to_boundary(n, torch.sort(line_points).values, device)

    def sample_grid(self, n=None, d=None, params=Points.empty(), device='cpu'):
        n = self.domain._compute_number_of_points(n, d, params)
        line_points = torch.linspace(0, self.domain.polygon.boundary.length,
                                     n+1, device=device)[:-1]
        return self._transform_points_to_boundary(n, line_points, device)

    def _transform_points_to_boundary(self, n, line_points, device):
        """Transform points that lay between 0 and polygon.boundary.length to 
        the surface of this polygon. The points have to be ordered from smallest
        to biggest.
        """
        outline = self.domain.outline(device=device)
        index = 0
        current_length = 0
        points = torch.zeros((n, 2), device=device)
        for boundary_part in outline:
            points, index, current_length = \
                self._distribute_line_to_boundary(points, index, line_points,
                                                  boundary_part, current_length)
        return Points(points, self.space)

    def _distribute_line_to_boundary(self, points, index, line_points,
                                     corners, current_length):
        corner_index = 0
        side_length = torch.linalg.norm(corners[1] - corners[0])
        while index < len(line_points):
            if line_points[index] <= current_length + side_length:
                point = self._translate_point_to_bondary(index, line_points,
                                                         corners, current_length,
                                                         corner_index, side_length)
                points[index] = point
                index += 1
            else:
                corner_index += 1
                current_length += side_length
                if corner_index >= len(corners) - 1:
                    break
                side_length = torch.linalg.norm(corners[corner_index+1]
                                             - corners[corner_index])
        return points, index, current_length

    def _translate_point_to_bondary(self, index, line_points, corners,
                                    current_length, corner_index, side_length):
        coord = line_points[index] - current_length
        new_point = (corners[corner_index] + coord/side_length *
                     (corners[corner_index+1] - corners[corner_index]))
        return new_point

    def normal(self, points, params=Points.empty(), device='cpu'):
        points, params, device = \
            self._transform_input_for_normals(points, params, device)
        points = points.as_tensor
        outline = self.domain.outline(device=device)
        index = self._where_on_boundary(points, outline)
        return self.normal_list[index].to(device)

    def _compute_normals(self, outline):
        face_number = sum([len(corners) for corners in outline])
        normal_list = torch.zeros((face_number, 2))
        index = 0
        for corners in outline:
            for i in range(len(corners)-1):
                normal = self._compute_local_normal_vector(corners[i+1], corners[i])
                normal_list[index] = normal
                index += 1
        return normal_list

    def _compute_local_normal_vector(self, end_corner, start_corner):
        conect_vector = end_corner - start_corner
        side_length = torch.linalg.norm(conect_vector, dim=0)
        normal = torch.index_select(conect_vector, 0, torch.LongTensor([1, 0]))
        normal /= side_length
        normal[1] *= -1
        return normal

    def _where_on_boundary(self, points, outline):
        index = -1 * torch.ones(len(points), dtype=int)
        counter = 0
        for corners in outline:
            for i in range(len(corners)-1):
                line = s_geo.LineString([corners[i], corners[i+1]])
                not_found = torch.where(index < 0)[0]
                for k in not_found:
                    point = s_geo.Point(points[k])
                    distance = line.distance(point)
                    if abs(distance) <= self.tol:
                        index[k] = counter
                counter += 1
        return index