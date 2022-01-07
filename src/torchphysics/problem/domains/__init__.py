"""DE domains built of a variety of geometries,
including functions to sample points etc"""

# 0D-domains:
from .domain0D.point import Point
# 1D-domains:
from .domain1D.interval import Interval
# 2D-domains:
from .domain2D.circle import Circle
from .domain2D.parallelogram import Parallelogram
from .domain2D.triangle import Triangle
#from .domain2D.shapely_polygon import ShapelyPolygon
# 3D-domains:
from .domain3D.sphere import Sphere 
#from .domain3D.trimesh_polyhedron import TrimeshPolyhedron