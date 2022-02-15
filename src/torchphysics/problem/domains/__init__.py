"""The domains package allows to define a variety of geometries. All geometries
allow to sample points to later train an artificial neural network with one
of the implemented physics approaches.

Boolean operations together with cartesian products and parameter-dependencies
form an easy-to-use toolbox enable the user to define a large set of relevant 
domains. In addition, domains can be imported from shapely or `.stl`-files (through
`trimesh`)."""

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