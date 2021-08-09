"""DE domains built of a variety of geometries,
including functions to sample points etc"""

from .domain import Domain
from .domain1D import Interval
from .domain2D import (Rectangle,
                       Circle,
                       Polygon2D,
                       Triangle)
from .domain3D import (Box, Sphere, Cylinder, Polygon3D)
from .domain_operations import (Cut, Union, Intersection)