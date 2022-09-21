"""Domains handle the geometries of the underlying problems. Every input variable, that 
appears in the differentialequation has to get a domain, to which it belongs.
Different 0D, 1D, 2D and 3D domains are pre implemented. For more complex domains four
operations are implemented:

- ``Union`` :math:`A \cup B`, implemented with: ``+``
- ``Intersection`` :math:`A \cap B`, implemented with: ``&``
- ``Cut`` :math:`A \setminus B`, implemented with: ``-``
- ``Cartesian product`` :math:`A \\times B`, implemented with: ``*``

It is possible to pass in functions as parameters of most domains. This leads to geometries that
can change depending on other variables, e.g. a moving domain in time. 

Boolean operations together with cartesian products and parameter-dependencies
form an easy-to-use toolbox enable the user to define a large set of relevant 
domains. In addition, domains can be imported from shapely or `.stl`-files (through
`trimesh`).

If you want to solve an inverse problem, the learnable parameters **do not** get a domain! They have to be 
defined with the torchphysics.model.Parameters class.
"""

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
# Function domains:
from .functionsets.functionset import FunctionSet, CustomFunctionSet
# Domain transforms:
from .domainoperations.translate import Translate
from .domainoperations.rotate import Rotate