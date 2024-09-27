import pytest
from math import sqrt
import numpy as np
from modulus.sym.geometry.geometry import Geometry
from modulus.sym.geometry.curve import Curve
from modulus.sym.geometry.parameterization import Parameterization
from torchphysics.wrapper.geometry import ParallelogramCylinder 


def test_ParallelogramCylinder_creation():
    origin = (0, 0, 0)
    corner1 = (1, 0, 0)
    corner2 = (0, 1, 0)
    height = 2
    parameterization = Parameterization()
    
    cylinder = ParallelogramCylinder(origin, corner1, corner2, height, parameterization)
    
    assert isinstance(cylinder, Geometry)    
    assert cylinder.parameterization == parameterization    
    assert callable(cylinder.sdf)
    assert cylinder.dims == ['x', 'y', 'z']    
    assert cylinder.sdf({'x': np.array([[0]]),'y':np.array([[0]]),'z':np.array([[0]])},params={})['sdf'][0][0] == 0
    assert cylinder.sdf({'x': np.array([[0.5]]),'y':np.array([[0.5]]),'z':np.array([[1]])},params={})['sdf'][0][0]==pytest.approx(0.5,rel=1e-12)
    assert cylinder.sdf({'x': np.array([[1.5]]),'y':np.array([[0.5]]),'z':np.array([[2]])},params={})['sdf'][0][0]== -0.5



def test_ParallelogramCylinder_curves():
    origin = (0, 0, 0)
    corner1 = (1, 0, 0)
    corner2 = (0, 1, 0)
    height = 2
    parameterization = Parameterization()
    
    cylinder = ParallelogramCylinder(origin, corner1, corner2, height, parameterization)
    
    assert len(cylinder.curves) == 6
    assert all(isinstance(curve, Curve) for curve in cylinder.curves)



def test_ParallelogramCylinder_invalid_parameters():
    origin = (0, 0, 0)
    corner1 = (1, 0, 1)  
    corner2 = (0, 1, 0)
    height = 2
    parameterization = Parameterization()
    
    with pytest.raises(AssertionError):
        ParallelogramCylinder(origin, corner1, corner2, height, parameterization)

def test_ParallelogramCylinder_negative_height():
    origin = (0, 0, 0)
    corner1 = (1, 0, 0)
    corner2 = (0, 1, 0)
    height = -2  # Negative height should raise an error or be considered invalid
    parameterization = Parameterization()
    
    with pytest.raises(TypeError):
        ParallelogramCylinder(origin, corner1, corner2, height, parameterization)




def test_ParallelogramCylinder_repr():
    origin = (0, 0, 0)
    corner1 = (1, 0, 0)
    corner2 = (0, 1, 0)
    height = 2
    parameterization = Parameterization()
    
    cylinder = ParallelogramCylinder(origin, corner1, corner2, height, parameterization)
    
    assert isinstance(repr(cylinder), str)
    assert "ParallelogramCylinder" in repr(cylinder)

