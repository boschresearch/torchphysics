# This file includes work covered by the following copyright and permission notices:
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
# Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import torch.nn as nn
from torch import atan2, sqrt
import numpy as np
from functools import reduce
from operator import mul

from modulus.sym.geometry.geometry import Geometry, csg_curve_naming
from modulus.sym.geometry.curve import Curve, SympyCurve
from modulus.sym.geometry.parameterization import Parameterization, Bounds, Parameter
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf

from sympy import Symbol, Min, Max, Heaviside, sqrt, Abs, sign, Eq, Or, And, Rational, lambdify
from sympy.vector import CoordSys3D


 
def GeometryAxischange(        
        self,
        axis1: str="x",
        axis2: str="y",
        axis3: str="z",
        parameterization=Parameterization(),
    ):
        """
        Exchange two axes of the geometry. Many primitive geometries in Modulus are defined in a specific coordinate system and this function allows to change the coordinate system of the geometry.
        It transforms the boundary curves, bounds and computes a new sdf function as wrapper function.        
        Parameters
        ----------
        axis1: str
            name of the first axis
        axis2: str
            name of the second axis
        axis3: str
            name of the third axis
        parameterization: Parameterization, optional
            parameterization of the geometry

        Returns
        -------
        Geometry
            new geometry object with exchanged axes

        """
        # create wrapper function for sdf function
        def _axischange_sdf(sdf, dims,axes):
            def axischange_sdf(invar, params, compute_sdf_derivatives=False):                
                changed_invar = {**invar}
                _changed_invar = {**changed_invar}
                for i, key in enumerate(dims):
                    _changed_invar[key] = changed_invar[axes[i]]
          
                # compute sdf
                computed_sdf = sdf(_changed_invar, params, compute_sdf_derivatives)
                return computed_sdf

            return axischange_sdf

        axes = (axis1,axis2,axis3)
        new_sdf = _axischange_sdf(self.sdf, self.dims, axes)
        # add parameterization
        new_parameterization = self.parameterization.union(parameterization)
        
        # change bounds
        bound_ranges = {**self.bounds.bound_ranges}
        _bound_ranges = {**bound_ranges}
        
        for i, key in enumerate(self.dims):             
            _bound_ranges[Parameter(key)] = bound_ranges[Parameter(axes[i])]
        
        new_bounds = self.bounds.copy()
        new_bounds.bound_ranges = _bound_ranges
        
        # change curves
        new_curves = []
        for c in self.curves:
            new_c = c.axischange(axes, parameterization)
            new_curves.append(new_c)
        
        # return rotated geometry
        return Geometry(
            new_curves,
            new_sdf,
            len(self.dims),
            new_bounds,
            new_parameterization,
            interior_epsilon=self.interior_epsilon,
        )

def CurveAxischange(self, axes, parameterization=Parameterization()):
        """
        Exchange two axes of the curves. Many primitive geometry curves 
        in Modulus are defined in a specific coordinate system and this 
        function allows to change the coordinate system of the curve.       
        It computes a new sample function for the curves as wrapper 
        function.        
        
        Parameters
        ----------
        axes: tuple
            names of the axes        
        parameterization: Parameterization, optional
            parameterization of the geometry
        Returns
        -------
        Curve
            new curve object with exchanged axes
        """
        def _sample(internal_sample, dims, axes):
            def sample(
                nr_points, parameterization=Parameterization(), quasirandom=False
            ):
                # sample points
                invar, params = internal_sample(
                    nr_points, parameterization, quasirandom
                )
                changed_invar = {**invar}
                _changed_invar = {**changed_invar}
                
                for i, key in enumerate(dims):
                    _changed_invar[key] = changed_invar[axes[i]]
                    _changed_invar['normal_'+key] = changed_invar['normal_'+axes[i]]
                
                return _changed_invar, params
                
            return sample
        return Curve(
            _sample(self._sample, self.dims,axes),
            len(self.dims),
            self.parameterization.union(parameterization),
        )    
    
# Modulus Curve class gets the additional method CurveAxischange
Curve.axischange=CurveAxischange
# Modulus Geometry class gets the additional method GeometryAxischange
Geometry.axischange=GeometryAxischange

class TPGeometryWrapper():
    """
    Wrapper to convert dictionary with spatial domain decomposition 
    into Modulus geometry. The domain is recursively analyzed mainly
    concerning domain operations and product domains to identify the 
    underlying spatial geometries.
    A general product of domains as in TorchPhysics can not be 
    implemented in Modulus, because the domain defines a spatial 
    geometry that must have a sdf implementation which is not available 
    in general for an arbitrary product domain.
    Therefore each product domain has to be analyzed and mapped on a 
    spatial geometry resulting on union/intersection/cut of a 
    translated/rotated primitive geometry of Modulus.

    Not supported types/combinations in TorchPhysics?
    
    Parameters
    ----------
    domain_dict: dictionary
        spatial domain decomposition
     
    
            
    """ 
    def __init__(self,domain_dict)-> None:
        self.domain_dict = domain_dict
    
    def getModulusGeometry(self):
        return self.RecursiveDomainDictGeoWrapper(self.domain_dict,translate_vec=[0,0,0],rotation_list=[[],[],[],0])

    
    def RecursiveDomainDictGeoWrapper(self,domain_dict,prod_factor=None,translate_vec=[0,0,0],rotation_list=[[],[],[],0]):
        """
        Recursive function that analyzes a dictionary of TorchPhysics 
        (spatial) domains as a result of domain decomposition to map 
        the domain to a Modulus geometry.
       
        Parameters
        ----------
        domain_dict: dictionary
            dictionary with TorchPhysics (spatial) domains. 
            Keys are domain operations:
                'u' (union), 'ub' (boundary of union),'d' (decomposed),
                'i' (intersection), 'ib' (boundary of intersection), 
                'c' (cut), 'cb' (boundary of cut), 't' (tranlation),
                'r' (rotation),'p' (product)
        prod_factor: float, optional
            domain factor of previous decomposed product (3D domain 
            can consist of two cross products)
        
        Returns
        -------
        geometry object
            Modulus geometry object
        is_boundary: bool
            True if domain is boundary
        cond: sympy expression
            condition to restrict on parts of geometry, resulting from 
            cross products
        translate_vec: list
            translation vector, containing the translation in x,y,z
        rotation_list: list
            list containing the rotation angles, rotation axes, rotation points and the priority of the rotation vs. translation            
        """
        
        for key, val in domain_dict.items():
                if (key=='u') or (key=='ub'):                       
                    is_boundary1, geometry1, cond1, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[0],prod_factor,translate_vec,rotation_list)   
                    is_boundary2, geometry2, cond2, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[1],prod_factor,translate_vec,rotation_list)                                          
                    
                    assert is_boundary1==is_boundary2
                    
                    if cond1==cond2:
                        if cond1!=None:
                            cond = cond1
                        else:
                            cond =None                       
                    elif cond1==None:
                        cond = cond2                 
                    elif cond2==None:
                        cond=cond1                        
                    else:          
                        cond = self._or_condition(self._geo_condition(geometry1.sdf,cond1,is_boundary1),self._geo_condition(geometry2.sdf,cond2,is_boundary2))                        
                    return is_boundary1,geometry1+geometry2, cond, translate_vec, rotation_list            
                elif key=='i' or key=='ib':
                    is_boundary1, geometry1, cond1, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[0],prod_factor,translate_vec, rotation_list)
                    is_boundary2, geometry2, cond2, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[1],prod_factor,translate_vec, rotation_list)
                    assert is_boundary1==is_boundary2
                    if cond1==cond2:
                        if cond1!=None:
                            cond = cond1
                        else:
                            cond =None                       
                    elif cond1==None:
                        cond = cond2                 
                    elif cond2==None:
                        cond=cond1                        
                    else:   
                        cond = "And(" + cond1 + "," + cond2 + ")" 
                    return is_boundary1, geometry1&geometry2, cond, translate_vec, rotation_list
               
                elif key=='c' or key=='cb':
                    is_boundary1, geometry1, cond1, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[0],prod_factor,translate_vec, rotation_list)
                    is_boundary2, geometry2, cond2, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[1],prod_factor,translate_vec, rotation_list)
                    assert is_boundary1==is_boundary2
                    if cond1==cond2:
                        if cond1!=None:
                            cond = cond1
                        else:
                            cond =None                       
                    elif cond1==None:
                        cond = cond2                 
                    elif cond2==None:
                        cond=cond1                        
                    else:   
                        cond = "Or(" + cond1 + "," + cond2 + ")"
                    return is_boundary1, geometry1-geometry2, cond, translate_vec, rotation_list                
                    
                elif key=='p':                      
                    if prod_factor is None:                    
                        return self.ProductDomainWrapper(val[0],val[1],translate_vec=translate_vec, rotation_list=rotation_list)  
                    else:
                        return self.ProductDomainWrapper(val[0],val[1],prod_factor,translate_vec=translate_vec, rotation_list=rotation_list)
            
               
                elif key=='t':   
                    assert ((next(iter(val[0].keys()))=='d') | (next(iter(val[0].keys()))=='r') ), "Only translation of primitive or rotated domains allowed"                                     
                    rot_angles, rot_axes, rot_points, rot_prio = rotation_list
                    if next(iter(val[0].keys()))=='d':
                        # translation has to be done first
                        rotation_list[3] = 1                        
                        dom_space = val[0]['d'].space
                    else:
                        # rotation has to be done first
                        rotation_list[3] = 0
                        assert (next(iter(val[0]['r'][0].keys()))=='d'), "Only translation of primitive or rotated domains allowed" 
                        dom_space = val[0]['r'][0]['d'].space
                   
                    vec = [float(val_in) for val_in in val[1]]                                        
                    translate_vec_new = self.compute_3D_translate_vec(dom_space,vec)
                    
                    if translate_vec !=[0,0,0]:
                        translate_vec =[translate_vec[ii]+translate_vec_new[ii] for ii in range(3)]
                    else:
                        translate_vec = translate_vec_new
                    
                    is_boundary, geometry, cond, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[0],prod_factor=prod_factor,translate_vec=translate_vec,rotation_list=rotation_list)                        
                    return is_boundary, geometry, cond, translate_vec, rotation_list
                elif key=='r':
                    assert ((next(iter(val[0].keys()))=='d') | (next(iter(val[0].keys()))=='t')), "Only rotation of primitive or translated domains allowed"                    
                    rot_angles, rot_axes, rot_points, rot_prio = rotation_list
                    

                    if next(iter(val[0].keys()))=='d':
                        # rotation has to be done first
                        rotation_list[3] = 0
                        dom_space = val[0]['d'].space   
                    else:
                        # translation has to be done first
                        rotation_list[3] = 1
                        assert (next(iter(val[0]['t'][0].keys()))=='d'), "Only rotation of primitive or translated domains allowed"   
                        dom_space = val[0]['t'][0]['d'].space
                                        
                    rot_matrix = val[1]                   
                    rot_point = [float(val_in) for val_in in val[2]]  
                    if len(rot_point)==2:                        
                        rot_axis, rot_point = self.compute_3D_rotation(dom_space,rot_point)                        
                        rot_angles.append(float(atan2(rot_matrix[0, 1, 0], rot_matrix[0, 0, 0])))
                        rot_axes.append(rot_axis)
                        rot_points.append(rot_point)
                    else: 
                        theta_x = float(atan2(rot_matrix[2, 1], rot_matrix[2, 2]))
                        theta_y = float(atan2(-rot_matrix[2, 0], sqrt((rot_matrix[2, 1] ** 2) + (rot_matrix[2, 2] ** 2))))
                        theta_z = float(atan2(rot_matrix[1, 0], rot_matrix[0, 0]))
                        rot_angles.extend([theta_x, theta_y, theta_z])
                        rot_axes.extend(['x', 'y', 'z'])
                        rot_points.extend([rot_point, rot_point, rot_point])
                    
                    is_boundary, geometry, cond, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(val[0],prod_factor=prod_factor,translate_vec=translate_vec,rotation_list=rotation_list)                        
                    return is_boundary, geometry, cond, translate_vec, rotation_list
                    
                elif key=='d':                   
                    if prod_factor is None:
                        return self.GeometryNameMapper(domain_dict['d'],translate_vec=translate_vec,rotation_list=rotation_list)
                    else:
                        return self.ProductDomainWrapper(domain_dict,prod_factor,translate_vec=translate_vec,rotation_list=rotation_list)
        
    
    def _geo_condition(self, sdf_func,sympy_expression,is_boundary):
        """
        Wrapper function to convert a general sympy_expression to a 
        condition that is only valid for points on the geometry 
        (interior or on the boundary).
        It is based on evaluating the sdf function of the geometry and 
        the sympy expression.
        It returns a condition function that can be evaluated for 
        arbitrary points, but is only true for points on the geometry 
        that satisfy the sympy expression.
        
        Parameters
        ----------
        sdf_func: function
            signed distance function
        sympy_expression: sympy expression
            sympy expression to restrict the geometry
        is_boundary: bool
            True if domain is boundary
        Returns
        -------
        function
            condition function
        """
        def geo_condition(invar,params):            
            x=Symbol('x')
            y=Symbol('y')
            z=Symbol('z')
            
            f2 = sdf_func(invar,params)
            # Get the number of arguments that f1 takes
            num_args = len(invar)

            if num_args == 3:
                # If f1 takes 3 arguments, call it with 'x', 'y', 'z'
                f1 = lambdify([(x,y,z)], sympy_expression)
                result = f1((invar['x'], invar['y'], invar['z']))
            elif num_args == 2:
                f1 = lambdify([(x,y)], sympy_expression)                
                result = f1((invar['x'], invar['y']))
            else:                
                f1 = lambdify([x], sympy_expression)
                result = f1(invar['x']) 
                        
            if is_boundary:
                return np.equal(f2["sdf"],0)&result  
            else:
                return np.greater(f2["sdf"],0)& result
        return geo_condition
    
    def _or_condition(self,f1,f2):
        """
        Wrapper function to combine two condition functions with an OR 
        operation.
        """        
        def or_condition(invar,params):
            return np.logical_or(f1(invar,params),f2(invar,params))
        return or_condition    

    def changeVarNames(self,domain,dim):       
        """
        Change variable names of TorchPhysics variables to Modulus 
        variable names, if variable 'x' is of higher dimension than 1.
        """
        vars = list(domain.space.keys())        
        if vars==['x']:
            if dim ==2:
                return ['x','y']
            elif dim ==3:
                return ['x','y','z']
            else:
                return ['x']
        else:
            return vars


    def ProductDomainWrapper(self,dom1,dom2,dom3=None,translate_vec = [0,0,0],rotation_list=[[],[],[],0]):    
        """
        Analyzes the elements of a decomposition of a TorchPhysics 
        product domain (cross product of domains) and maps the whole 
        product domain to a Modulus geometry.
       
        Parameters
        ----------
        dom1: domain object
            domain object from TorchPhysics
        dom2: domain object
            domain object from TorchPhysics
        dom3: domain object, optional
            domain object from TorchPhysics
        
        Returns
        -------
        geometry object
            Modulus geometry object        
        is_boundary: bool
            True if domain is boundary
        cond: sympy expression
            condition to restrict on parts of geometry
        """                  
        imported_module_3d = importlib.import_module("modulus.sym.geometry.primitives_3d")
        imported_module_2d = importlib.import_module("modulus.sym.geometry.primitives_2d")         
        
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        cond=None
        if dom3 == None:           
            key1 = next(iter(dom1.keys()))
            val1 = dom1[key1]
            key2 = next(iter(dom2.keys()))    
            val2 = dom2[key2]           
        
            # sorting by domain dimension: dom1.dim>=dom2.dim ?
            # 2D geometries in 3D define boundaries!            
            if key1 == 'd':
                if key2 == 'd':
                    if val1.dim < val2.dim:
                        dom1,dom2 = dom2,dom1
                        key1 = next(iter(dom1.keys()))
                        val1 = dom1[key1]
                        key2 = next(iter(dom2.keys()))    
                        val2 = dom2[key2]
                    var1 = self.changeVarNames(val1,val1.dim)
                    var2 = self.changeVarNames(val2,val2.dim)
                    # Cross product of 2D domain and point / interval boundary    
                    if (val1.dim==2) & (val2.dim==0): 
                        if var2[0] == 'z':
                            if var1 == ['x','y']:
                                varchange = False
                            else:
                                var1=['x','y']                    
                                varchange = True
                        elif var2[0] == 'x':
                            if var1 == ['z','y']:
                                varchange = False
                            else:
                                var1=['z','y']                    
                                varchange = True
                        else:
                            if var1 == ['x','z']:
                                varchange = False
                            else:
                                var1=['x','z']                    
                                varchange = True
                                            
                        is_boundary = True
                        #only single boundary point
                        if hasattr(val2,'side'):                        
                            point1=float(val2.side())
                            point2=point1+1
                            cond=Eq(Symbol(var2[0]),Rational(point1))   
                        elif (type(val2).__name__=='Point'):
                            point1=float(val2.point())
                            point2=point1+1
                            cond=Eq(Symbol(var2[0]),Rational(point1))   
                        else:
                            point1=float(val2.bounding_box()[0])
                            point2=float(val2.bounding_box()[1])                           
                            cond=Or(Eq(Symbol(var2[0]),Rational(point1)),Eq(Symbol(var2[0]),Rational(point2)))
                        if type(val1).__name__ == 'Parallelogram':                               
                            orig=self.varChange(tuple(element.item() for element in val1.origin()),varchange)
                            c1=self.varChange(tuple(element.item() for element in val1.corner_1()),varchange)
                            c2=self.varChange(tuple(element.item() for element in val1.corner_2()),varchange)                           
                            geometry=ParallelogramCylinder(orig+(point1,),c1+(point1,),c2+(point1,),height=point2-point1).axischange(var1[0],var1[1],var2[0])
                        elif type(val1).__name__ == 'Circle':                              
                            center = self.varChange(tuple(element.item() for element in val1.center()),varchange)
                            radius = float(val1.radius())
                            height = point2-point1                        
                            geometry=getattr(imported_module_3d,"Cylinder")(center+(point1+height/2,),radius,height).axischange(var1[0],var1[1],var2[0])                            
                        elif type(val1).__name__ == 'Triangle': 
                            #only Isosceles triangle with axis of symmetry parallel to y-axis
                            assert (val1.origin()[1]==val1.corner_1()[1]), "Symmetry axis of triangle has to be y-axis parallel!"
                            assert (np.linalg.norm(val1.corner_2()-val1.origin()) == np.linalg.norm(val1.corner_2()-val1.corner_1())), "Triangle not Isosceles!"
                            base = float(val1.corner_1()[0]-val1.origin()[0])
                            height = float(np.sqrt(np.linalg.norm(val1.corner_2()-val1.corner_1())**2-(base/2)**2))
                            height_prism = float(point2-point1)
                            center = self.varChange((float(val1.origin()[0])+base/2,float(val1.origin()[1])),varchange)
                            geometry=getattr(imported_module_3d,"IsoTriangularPrism")(center+(point1+height_prism/2,),base,height,height_prism).axischange(var1[0],var1[1],var2[0])
                        else:                
                            assert False, "Type of product domain not supported: "+type(val1).__name__+" * "+type(val2).__name__                  
                    # Cross product of 2D domain and interval        
                    elif (val1.dim==2) & (val2.dim==1):                       
                        if var2[0] == 'z':
                            if var1 == ['x','y']:
                                varchange = False
                            else:
                                var1=['x','y']                    
                                varchange = True
                        elif var2[0] == 'x':
                            if var1 == ['z','y']:
                                varchange = False
                            else:
                                var1=['z','y']                    
                                varchange = True
                        else:
                            if var1 == ['x','z']:
                                varchange = False
                            else:
                                var1=['x','z']                    
                                varchange = True
                       
                        assert (type(val2).__name__ == 'Interval'), "Type of product domain not supported: Should be 2D domain * Interval"
                        
                        is_boundary = False                    
                        point1=float(val2.bounding_box()[0])
                        point2=float(val2.bounding_box()[1])
                        
                        if type(val1).__name__ == 'Parallelogram':                               
                            orig= self.varChange(tuple(element.item() for element in val1.origin()),varchange)
                            c1= self.varChange(tuple(element.item() for element in val1.corner_1()),varchange)
                            c2= self.varChange(tuple(element.item() for element in val1.corner_2()) ,varchange)                          
                            geometry=ParallelogramCylinder(orig+(point1,),c1+(point1,),c2+(point1,),height=point2-point1).axischange(var1[0],var1[1],var2[0])
                        elif type(val1).__name__ == 'Circle':                              
                            center =  self.varChange(tuple(element.item() for element in val1.center()),varchange)
                            radius = float(val1.radius())       
                            height = point2-point1
                            geometry=getattr(imported_module_3d,"Cylinder")(center+(point1+height/2,),radius,height).axischange(var1[0],var1[1],var2[0])                            
                        elif type(val1).__name__ == 'Triangle': 
                            #only Isosceles triangle with axis of symmetry parallel to y-axis
                            assert (val1.origin()[1]==val1.corner_1()[1]),"Symmetry axis of triangle has to be y-axis parallel!"
                            assert (np.linalg.norm(val1.corner_2()-val1.origin()) == np.linalg.norm(val1.corner_2()-val1.corner_1())), "Triangle not Isosceles!"
                            base = float(val1.corner_1()[0]-val1.origin()[0])
                            height = float(np.sqrt(np.linalg.norm(val1.corner_2()-val1.corner_1())**2-(base/2)**2))
                            height_prism = point2-point1                            
                            center = self.varChange((float(val1.origin()[0])+base/2,float(val1.origin()[1])),varchange)                            
                            geometry=getattr(imported_module_3d,"IsoTriangularPrism")(center+(point1+height_prism/2,),base,height,height_prism).axischange(var1[0],var1[1],var2[0])
                        else:                              
                            assert False, "Type of product domain not supported: "+type(val1).__name__+" * "+type(val2).__name__                        
                    # Cross product of 1D domain (boundary of 2D domain) and interval
                    elif (val1.dim==1) & (val2.dim==1):
                        assert ((type(val2).__name__ == 'Interval')|(type(val1).__name__ == 'Interval')), "Product domain for these types of domain not allowed: Has to be 1D domain * Interval"                        
                        is_boundary = True 
                        if type(val2).__name__ != 'Interval':
                            val1,val2=val2,val1                            
                            var1 = self.changeVarNames(val1,val1.dim)
                            var2 = self.changeVarNames(val2,val2.dim)
                        if var2[0] == 'z':
                            if var1 == ['x','y']:
                                varchange = False
                            else:
                                var1=['x','y']                    
                                varchange = True
                        elif var2[0] == 'x':
                            if var1 == ['z','y']:
                                varchange = False
                            else:
                                var1=['z','y']                    
                                varchange = True
                        else:
                            if var1 == ['x','z']:
                                varchange = False
                            else:
                                var1=['x','z']                    
                                varchange = True                                               
                           
                        point1=float(val2.bounding_box()[0])
                        point2=float(val2.bounding_box()[1])                        
                        cond=And(Symbol(var2[0])<Rational(point2),Symbol(var2[0])>Rational(point1))
                        if type(val1).__name__ == 'ParallelogramBoundary':                               
                            orig= self.varChange(tuple(element.item() for element in val1.domain.origin()),varchange)
                            c1= self.varChange(tuple(element.item() for element in val1.domain.corner_1()),varchange)
                            c2= self.varChange(tuple(element.item() for element in val1.domain.corner_2()) ,varchange)                          
                            geometry=ParallelogramCylinder(orig+(point1,),c1+(point1,),c2+(point1,),height=point2-point1).axischange(var1[0],var1[1],var2[0])
                        elif type(val1).__name__ == 'CircleBoundary':                              
                            center =  self.varChange(tuple(element.item() for element in val1.domain.center()),varchange)
                            radius = float(val1.domain.radius())
                            height = point2-point1
                            geometry=getattr(imported_module_3d,"Cylinder")(center+(point1+height/2,),radius,height).axischange(var1[0],var1[1],var2[0])                                                                        
                        elif type(val1).__name__ == 'TriangleBoundary': 
                           #only Isosceles triangle with axis of symmetry parallel to y-axis
                            assert (val1.domain.origin()[1]==val1.domain.corner_1()[1]), "Symmetry axis of triangle has to be y-axis parallel!"
                            assert (np.linalg.norm(val1.domain.corner_2()-val1.domain.origin()) == np.linalg.norm(val1.domain.corner_2()-val1.domain.corner_1())), "Triangle not Isosceles!"
                            base = float(val1.domain.corner_1()[0]-val1.domain.origin()[0])
                            height = float(np.sqrt(np.linalg.norm(val1.domain.corner_2()-val1.domain.corner_1())**2-(base/2)**2))
                            height_prism = point2-point1
                            center =  self.varChange((float(val1.domain.origin()[0])+base/2,float(val1.domain.origin()[1])),varchange)
                            geometry=getattr(imported_module_3d,"IsoTriangularPrism")((center+(point1+height_prism/2,),base,height,height_prism).axischange(var1[0],var1[1],var2[0]))
                        elif type(val1).__name__ == 'Interval':
                            #var1 = list(val1.space.keys())
                            #var2 = list(val2.space.keys()) 
                            var1 = self.changeVarNames(val1,val1.dim)
                            var2 = self.changeVarNames(val2,val2.dim)
                            assert (((var1[0]=='x')&(var2[0]=='y'))|((var1[0]=='y')&(var2[0]=='x'))), "Only x,y as coordinates for 2D problems allowed"
                            
                            point1=float(val1.bounding_box()[0])
                            point2=float(val1.bounding_box()[1])
                            point3=float(val2.bounding_box()[0])
                            point4=float(val2.bounding_box()[1]) 
                            geometry=getattr(imported_module_2d,"Rectangle")((point1,point3),(point2,point4)).axischange(var1[0],var2[0],[])
                            cond = None
                            is_boundary=False
                        else:  
                            assert False, "Type of product domain not supported: "+type(val1).__name__+" * "+type(val2).__name__
                        
                    # Cross product of 1D domain (boundary of 2D domain or Interval) and point/boundary of interval
                    elif (val1.dim==1) & (val2.dim==0):
                        is_boundary = True                                         
                        if var2[0] == 'z':
                            if var1 == ['x','y']:
                                varchange = False
                            else:
                                var1=['x','y']                    
                                varchange = True
                        elif var2[0] == 'x':
                            if var1 == ['z','y']:
                                varchange = False
                            else:
                                var1=['z','y']                    
                                varchange = True
                        else:
                            if var1 == ['x','z']:
                                varchange = False
                            else:
                                var1=['x','z']                    
                                varchange = True
                                            
                        
                        if hasattr(val2,'side'):                         
                            point1=float(val2.side())
                            point2=point1+1
                            cond=Eq(Symbol(var2[0]),Rational(point1))
                        elif (type(val2).__name__=='Point'):
                            point1=float(val2.point())
                            point2=point1+1
                            cond=Eq(Symbol(var2[0]),Rational(point1)) 
                        else:
                            point1=float(val2.bounding_box()[0])
                            point2=float(val2.bounding_box()[1])                        
                            cond=Or(Eq(Symbol(var2[0]),Rational(point1)),Eq(Symbol(var2[0]),Rational(point2)))                                            
                        
                        if type(val1).__name__ == 'ParallelogramBoundary':                               
                            orig=self.varChange(tuple(element.item() for element in val1.domain.origin()),varchange)
                            c1=self.varChange(tuple(element.item() for element in val1.domain.corner_1()),varchange)
                            c2=self.varChange(tuple(element.item() for element in val1.domain.corner_2()) ,varchange)                          
                            geometry=ParallelogramCylinder(orig+(point1,),c1+(point1,),c2+(point1,),height=point2-point1).axischange(var1[0],var1[1],var2[0])
                        elif type(val1).__name__ == 'CircleBoundary':                              
                            center = self.varChange(tuple(element.item() for element in val1.domain.center()),varchange)
                            radius = float(val1.domain.radius())
                            height = point2-point1
                            geometry=getattr(imported_module_3d,"Cylinder")(center+(point1+height/2,),radius,height).axischange(var1[0],var1[1],var2[0])                                                                        
                        elif type(val1).__name__ == 'TriangleBoundary': 
                            #only Isosceles triangle with axis of symmetry parallel to y-axis
                            assert (val1.domain.origin()[1]==val1.domain.corner_1()[1]),"Symmetry axis of triangle has to be y-axis parallel!"
                            assert (np.linalg.norm(val1.domain.corner_2()-val1.domain.origin()) == np.linalg.norm(val1.domain.corner_2()-val1.domain.corner_1())), "Triangle not Isosceles!"
                            base = float(val1.domain.corner_1()[0]-val1.domain.origin()[0])
                            height = float(np.sqrt(np.linalg.norm(val1.domain.corner_2()-val1.domain.corner_1())**2-(base/2)**2))
                            height_prism = point2-point1
                            center = self.varChange((float(val1.domain.origin()[0])+base/2,float(val1.domain.origin()[1])),varchange)
                            geometry=getattr(imported_module_3d,"IsoTriangularPrism")(center+(point1+height_prism/2,),base,height,height_prism).axischange(var1[0],var1[1],var2[0])
                        elif type(val1).__name__ == 'Interval':
                            var1 = list(val1.space.keys())
                            var2 = list(val2.space.keys()) 
                            assert ((var1[0]!='z')&(var2[0]!='z')), "Only x,y as coordinates for 2D problems allowed"
                            point3=float(val1.bounding_box()[0])
                            point4=float(val1.bounding_box()[1])                        
                            geometry=getattr(imported_module_2d,"Rectangle")((point3,point1),(point4,point2)).axischange(var1[0],var2[0],[])                        
                        else:  
                            assert False, "Type of product domain not supported: "+type(val1).__name__+" * "+type(val2).__name__ 
                    # Cross product of two 0D domains not allowed
                    elif (val1.dim==0) & (val2.dim==0):                  
                        assert(False), "2D or 3D points are not allowed for sampling due to zero surface"
                else:
                    is_boundary, geometry, cond, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(dom2,dom1,translate_vec=translate_vec,rotation_list=rotation_list)
            else:                
                is_boundary, geometry, cond, translate_vec, rotation_list = self.RecursiveDomainDictGeoWrapper(dom1,dom2,translate_vec=translate_vec,rotation_list=rotation_list)
        else: # dom3 != None
            key1 = next(iter(dom1.keys()))
            val1 = dom1[key1]
            key2 = next(iter(dom2.keys()))    
            val2 = dom2[key2]
            key3 = next(iter(dom3.keys()))
            val3 = dom3[key3]
                        
            # fully decomposed product domain
            if (key1 == 'd')& (key2 == 'd')&(key3=='d'):
                # I1xI2xI3  I1xI2xP/IBoundary
                # Edges and 3D points are not allowed
                # sort by (x,y,z)-order
                vals= sorted((val1,val2,val3), key=lambda x: list(x.space.variables)[0])
                assert (np.sum([vals[0].dim,vals[1].dim,vals[2].dim])>1), "3D points or edges are not allowed due to zero surface in 3D"
                vars = ['x','y','z']                              
                points=[]
                conds=[]
                is_boundary = False
                for index, val in enumerate(vals):                  
                    if hasattr(val,'side'):                         
                        points.append([float(val.side()),float(val.side())+1])
                        conds.append(Eq(Symbol(vars[index]),Rational(float(val.side()))))
                        is_boundary = True                        
                    elif (type(val).__name__=='Point'):                                           
                        points.append([float(val.point()),float(val.point())+1])                         
                        conds.append(Eq(Symbol(vars[index]),Rational(float(val.point()))))
                        is_boundary = True
                    elif (type(val).__name__=='Interval'):
                        points.append([float(val.bounding_box()[0]),float(val.bounding_box()[1])])
                        conds.append(True)                    
                    else: # IntervalBoundary with both sides
                        points.append([float(val.bounding_box()[0]),float(val.bounding_box()[1])])
                        conds.append(Or(Eq(Symbol(vars[index]),Rational(float(val.bounding_box()[0]))),Eq(Symbol(vars[index]),Rational(float(val.bounding_box()[1])))))
                        is_boundary=True
                geometry=getattr(imported_module_3d,"Box")((points[0][0],points[1][0],points[2][0]),(points[0][1],points[1][1],points[2][1]))                   
                cond_all = And(conds[0],conds[1],conds[2])
                cond = None if cond_all == True else cond_all
            # further decomposition of product domain, but only union or union boundary of 1D or 0D domains allowed, that means no further rotation and translation will be done by call of ProductDomainWrapper
            else:
                #sort domains: domains with key other than 'd' first
                doms,keys = zip(*sorted(list(zip([dom1,dom2,dom3],[key1,key2,key3])), key=lambda x: x[1] == 'd'))                
                #only union or union boundary of 1D or 0D domains allowed (key='u' or 'ub')
                assert((keys[0]=='u')|(keys[0]=='ub')), "Other domain operations than Union or UnionBoundary not allowed"
                
                is_boundary1, geometry1, cond1, translate_vec, rotation_list = self.ProductDomainWrapper(doms[0][keys[0]][0],doms[1],doms[2],translate_vec, rotation_list)
                is_boundary2, geometry2, cond2, translate_vec, rotation_list = self.ProductDomainWrapper(doms[0][keys[0]][1],doms[1],doms[2],translate_vec, rotation_list)
                #sampler (interior or boundary) has to be the same for both domains, union of interior and boundary is not allowed
                assert is_boundary1==is_boundary2
                
                if cond1==cond2:
                    if cond1!=None:
                        cond = cond1
                    else:
                        cond =None
                       
                elif cond1==None:
                    cond = cond2                 
                elif cond2==None:
                    cond=cond1
                else:                        
                    cond = self._or_condition(self._geo_condition(geometry1.sdf,cond1,is_boundary1),self._geo_condition(geometry2.sdf,cond2,is_boundary2))   
                is_boundary = is_boundary1 
                geometry = geometry1+geometry2
                
                
        # check if rotation and translation have to be done
        rot_angles, rot_axes, rot_points, rot_prio = rotation_list
        if rot_angles != []:
            if translate_vec != [0,0,0]:
                if rot_prio == 0:
                    for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                        geometry = geometry.rotate(angle,axis,point)                                                          
                    geometry = geometry.translate(translate_vec) 
                    cond = self.translate_condition(cond,translate_vec)                    
                else:
                    geometry = geometry.translate(translate_vec) 
                    cond = self.translate_condition(cond,translate_vec)                   
                    for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                        geometry = geometry.rotate(angle,axis,point)                        
                rotation_list = [[],[],[],0]
                translate_vec = [0,0,0]
            else:
                for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                    geometry = geometry.rotate(angle,axis,point)                               
                rotation_list = [[],[],[],0]
        else:
            if translate_vec != [0,0,0]:
                geometry = geometry.translate(translate_vec) 
                cond = self.translate_condition(cond,translate_vec) 
                translate_vec = [0,0,0]
                
        return is_boundary, geometry, cond, translate_vec, rotation_list
            
        
   
    def varChange(self,val,ischange):
        """
        Convert 2D numpy array to tuple of floats and change the order 
        of the values if ischange is True.

        Parameters
        ----------
        val: numpy array
            array with 2 values
        ischange: bool
            change the order of the values
        
        Returns
        -------
        tuple
            tuple of floats
        """
        if ischange:
            return (float(val[1]), float(val[0]))
        else:       
            return (float(val[0]), float(val[1]))

    def GeometryNameMapper(self,domain,translate_vec=[0,0,0],rotation_list=[[],[],[],0]):
        """
        Mapping of single TorchPhysics (decomposed) domain to Modulus 
        geometry

        Parameters
        ----------
        domain: domain object
            domain object from TorchPhysics
        
        Returns
        -------
        geometry object
            Modulus geometry object 
        is_boundary: bool
            True if domain is boundary
        cond: sympy expression
            condition to restrict on parts of geometry
        """
        
        if 'Boundary' in type(domain).__name__:
            dim = domain.dim+1
            is_boundary = True
        else:
            dim = domain.dim
            is_boundary = False
        cond = None 
        # if variable x with higher dimension than 1, change to x,y (,z)
        vars = self.changeVarNames(domain,dim)               
        if dim ==1:            
            imported_module = importlib.import_module("modulus.sym.geometry.primitives_1d")
            if type(domain).__name__ == 'Interval': # IntervalBoundary                
                geometry = getattr(imported_module,"Line1D")(float(domain.lower_bound()),float(domain.upper_bound()))
            elif type(domain).__name__ == 'IntervalBoundary':                
                geometry = getattr(imported_module,"Line1D")(float(domain.domain.lower_bound()),float(domain.domain.upper_bound()))
            elif type(domain).__name__ == 'IntervalSingleBoundaryPoint':
                geometry = getattr(imported_module,"Line1D")(float(domain.domain.lower_bound()),float(domain.domain.upper_bound()))
                cond = Eq(Symbol('x'),Rational(float(domain.side())))                
            else:
                assert False, "Domain type not supported"
        elif dim==2: 
            if vars == ['x','y']:
                varchange = False
                y_index =1
            else:                
                varchange = True   
                y_index = 0                    
            imported_module = importlib.import_module("modulus.sym.geometry.primitives_2d")                     
            if (type(domain).__name__ == 'Parallelogram'):       
                origin = self.varChange(domain.origin(),varchange)
                corner1 = self.varChange(domain.corner_1(),varchange)
                corner2 = self.varChange(domain.corner_2(),varchange)                                         
                geometry = getattr(imported_module,"Polygon")([origin,corner1, np.subtract(np.add(corner1,corner2),origin),corner2])
            elif (type(domain).__name__ == 'ParallelogramBoundary'):                
                origin = self.varChange(domain.domain.origin(),varchange)
                corner1 = self.varChange(domain.domain.corner_1(),varchange)
                corner2 = self.varChange(domain.domain.corner_2(),varchange)                  
                geometry = getattr(imported_module,"Polygon")([origin,corner1, np.subtract(np.add(corner1,corner2),origin),corner2])
            elif (type(domain).__name__ == 'Circle'): 
                center = self.varChange(domain.center(),varchange)
                geometry = getattr(imported_module,"Circle")(center,float(domain.radius()))
            elif (type(domain).__name__ == 'CircleBoundary'):  
                center = self.varChange(domain.domain.center(),varchange)       
                geometry = getattr(imported_module,"Circle")(center,float(domain.domain.radius()))  
            elif (type(domain).__name__ == 'ShapelyPolygon'):                       
                geometry = getattr(imported_module,"Polygon")(list(zip(*domain.polygon.exterior.coords.xy))[:-1])                
            elif (type(domain).__name__ == 'ShapelyBoundary'):         
                geometry = getattr(imported_module,"Polygon")(list(zip(*domain.domain.polygon.exterior.coords.xy))[:-1])                
            elif (type(domain).__name__ == 'Triangle'):                 
                #only Isosceles triangle with axis of symmetry parallel to y-axis                 
                assert (domain.origin()[y_index]==domain.corner_1()[y_index]),"Symmetry axis of triangle has to be y-axis parallel!"
                assert (np.linalg.norm(domain.corner_2()-domain.origin()) == np.linalg.norm(domain.corner_2()-domain.corner_1())), "Triangle not Isosceles!"
                origin = self.varChange(domain.origin(),varchange)
                corner1 = self.varChange(domain.corner_1(),varchange)
                corner2 = self.varChange(domain.corner_2(),varchange)
                base = float(corner1[0]-origin[0])                           
                height = float(np.sqrt(np.linalg.norm(np.subtract(corner2,corner1))**2-(base/2)**2))                                
                center = (float(origin[0])+base/2,float(origin[1]))                
                geometry=getattr(imported_module,"Triangle")(center,base,height)    
            elif (type(domain).__name__ == 'TriangleBoundary'):                         
                #only Isosceles triangle with axis of symmetry parallel to y-axis
                assert (domain.domain.origin()[y_index]==domain.domain.corner_1()[y_index]), "Symmetry axis of triangle has to be y-axis parallel!"
                assert (np.linalg.norm(domain.domain.corner_2()-domain.domain.origin()) == np.linalg.norm(domain.domain.corner_2()-domain.domain.corner_1())), "Triangle not Isosceles!"                
                origin = self.varChange(domain.domain.origin(),varchange)
                corner1 = self.varChange(domain.domain.corner_1(),varchange)
                corner2 = self.varChange(domain.domain.corner_2(),varchange)
                base = float(corner1[0]-origin[0])                
                height = float(np.sqrt(np.linalg.norm(np.subtract(corner2,corner1))**2-(base/2)**2))                                
                center = (float(origin[0])+base/2,float(origin[1]))                
                geometry=getattr(imported_module,"Triangle")(center,base,height)
            else: 
                assert (True), "Domain type not supported"  
        else: #dim==3            
            imported_module = importlib.import_module("modulus.sym.geometry.primitives_3d")
            if type(domain).__name__ == 'Sphere':                   
                    _, sorted_center = zip(*(sorted(zip(vars,domain.center())))) 
                    sorted_center = tuple([float(elem) for elem in sorted_center])                                                
                    geometry = getattr(imported_module,"Sphere")(sorted_center,float(domain.radius()))
            elif type(domain).__name__ == 'SphereBoundary':        
                    _, sorted_center = zip(*(sorted(zip(vars,domain.domain.center())))) 
                    sorted_center = tuple([float(elem) for elem in sorted_center])                            
                    geometry = getattr(imported_module,"Sphere")(sorted_center,float(domain.domain.radius()))    
            elif type(domain).__name__ == 'TrimeshPolyhedron':
                try:
                    geometry = getattr(importlib.import_module("modulus.sym.geometry.tessellation"),"Tessellation")(domain.mesh)
                except:                    
                    raise Exception("Tessellation module only supported for Modulus docker installation due to missing pysdf installation!")
            elif type(domain).__name__ == 'TrimeshBoundary':
                try:
                    geometry = getattr(importlib.import_module("modulus.sym.geometry.tessellation"),"Tessellation")(domain.domain.mesh)
                except:
                    raise Exception("Tessellation module only supported for Modulus docker installation due to missing pysdf installation!")
            else: 
                assert (True), "Domain type not supported"  
        
        # check if rotation and translation have to be done
        rot_angles, rot_axes, rot_points, rot_prio = rotation_list
        if rot_angles != []:
            if translate_vec != [0,0,0]:
                if rot_prio == 0:
                    for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                        geometry = geometry.rotate(angle,axis,point)                        
                    geometry = geometry.translate(translate_vec) 
                    cond = self.translate_condition(cond,translate_vec)                    
                else:
                    geometry = geometry.translate(translate_vec) 
                    cond = self.translate_condition(cond,translate_vec)                   
                    for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                        geometry = geometry.rotate(angle,axis,point)                        
                rotation_list = [[],[],[],0]
                translate_vec = [0,0,0]
            else:
                for angle, axis, point in zip(rot_angles,rot_axes,rot_points):
                    geometry = geometry.rotate(angle,axis,point)                    
                rotation_list = [[],[],[],0]
        else:
            if translate_vec != [0,0,0]:
                geometry = geometry.translate(translate_vec) 
                cond = self.translate_condition(cond,translate_vec) 
                translate_vec = [0,0,0]

        return is_boundary, geometry, cond, translate_vec, rotation_list


    def translate_condition(self,cond,translate_vec):
        """
        Translate condition by vector

        Parameters
        ----------
        cond: sympy expression
            condition to translate
        translate_vec: tuple of floats
            translation vector
        
        Returns
        -------
        sympy expression
            translated condition
        """
        if cond is not None:
            return cond.subs(Symbol('x'),Symbol('x')-Rational(translate_vec[0])).subs(Symbol('y'),Symbol('y')-Rational(translate_vec[1])).subs(Symbol('z'),Symbol('z')-Rational(translate_vec[2]))
        else:
            return None

    def compute_3D_translate_vec(self,space,translate_vec):
        """
        Compute 3D translation vector in the order "x","y","z" out of 
        1D, 2D or 3D translation vector

        Parameters
        ----------
        space: dict
        translate_vec: tuple of floats
            translation vector
        
        Returns
        -------
        tuple of floats
            translated geometry
        """
        vec= [0,0,0]
        if 'x' in space:
            if space['x'] ==2:
                vec= [translate_vec[0],translate_vec[1],0]
            elif space['x'] ==3:
                vec= translate_vec
            else:
                for ind,key in enumerate(space.keys()):
                    if key =='x':
                        vec[0]=translate_vec[ind]
                    elif key =='y':
                        vec[1]=translate_vec[ind]
                    elif key =='z':
                        vec[2]=translate_vec[ind]
        else:
            for ind,key in enumerate(space.keys()):                  
                if key =='y':
                    vec[1]=translate_vec[ind]
                elif key =='z':
                    vec[2]=translate_vec[ind]    

        return vec
    
    def compute_3D_rotation(self,space,point):
        """
        Compute 3D rotation point in the order "x","y","z" out of 2D 
        rotation point and also rotation axis

        Parameters
        ----------
        space: dict
            space of domain (variables "x","y" or "x","z" or "y","z" or 
            "x" (2D))
        point: tuple of floats (2D)
        
        Returns
        -------
        rot_ax: str
            rotation axis
        rot_point: tuple of floats
            rotation point
        """
        rot_point = [0.0,0.0,0.0]
        if 'x' in space:
            if space['x'] ==2:
                rot_point= [point[0],point[1],0.0]
                rot_ax = 'z'            
            
            else:
                for ind,key in enumerate(space.keys()):
                    if key =='x':
                        rot_point[0]=point[ind]
                    elif key =='y':
                        rot_point[1]=point[ind] 
                        rot_point[2]=0.0
                        rot_ax = 'z'                                               
                    elif key =='z':
                        rot_point[2]=point[ind]
                        rot_point[1]=0.0
                        rot_ax = 'y'
        else:
            for ind,key in enumerate(space.keys()):                  
                if key =='y':
                    rot_point[1]=point[ind] 
                elif key =='z':
                    rot_point[2]=point[ind]
            rot_point[0]=0.0
            rot_ax = 'x'        

        return rot_ax, rot_point

class ParallelogramCylinder(Geometry):
    """
    3D Cylinder with Parallelogram base area perpendicular to z-axis

    Parameters
    ----------
    origin, corner1,corner2 : tuples with 3 ints or floats        
        Three corners of the parallelogram, in the following order

        |       corner_2 -------- x
        |      /                 /
        |     /                 /
        |    origin ----- corner_1
    height: z-coordinate of upper plane of parallelogram
    parameterization : Parameterization
        Parameterization of geometry.


    """

    def __init__(self, origin,corner1,corner2,height, parameterization=Parameterization()):
        assert ((origin[2] == corner1[2]) & (origin[2] == corner2[2])), "Points must have same coordinate on normal dim:z"

        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        
        s1, s2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))       
       

        # surface
        curve_parameterization = Parameterization({s1: (0, 1),s2: (0,1)})
        curve_parameterization = Parameterization.combine(curve_parameterization, parameterization)
                
        # area
        N = CoordSys3D("N")        
        vec1 = tuple(np.subtract(corner1,origin))
        vec2 = tuple(np.subtract(corner2,origin))
        vvec1 = vec1[0]*N.i + vec1[1]*N.j + vec1[2]*N.k
        vvec2 = vec2[0]*N.i + vec2[1]*N.j + vec2[2]*N.k
        
        corner3 = tuple(np.subtract(np.add(corner1,corner2),origin))
               
        #compute cross product of the "base"-vectors for computing area and testing of right-handed system
        cross_vec = vvec1.cross(vvec2)
        sgn_normal = sign(cross_vec.dot(N.k))
        
        area_p= sqrt(cross_vec.dot(cross_vec))
        
               
        bottom = SympyCurve(
                functions={
                    "x": origin[0]+ s1*vec1[0]+s2*vec2[0],
                    "y": origin[1]+ s1*vec1[1]+s2*vec2[1],
                    "z": origin[2],
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": -1,
                },
                parameterization=curve_parameterization,
                area=area_p,
            )
        top = SympyCurve(
                functions={
                    "x": origin[0]+ s1*vec1[0]+s2*vec2[0],
                    "y": origin[1]+ s1*vec1[1]+s2*vec2[1],
                    "z": origin[2]+height,
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": 1,
                },
                parameterization=curve_parameterization,
                area=area_p,
            )
        norm_l=np.linalg.norm([-vec2[1],vec2[0]])
        side1 = SympyCurve(
                functions={
                    "x": origin[0]+ s1*vec2[0],
                    "y": origin[1]+ s1*vec2[1],
                    "z": origin[2]+ s2*height,
                    "normal_x": -vec2[1]/norm_l*sgn_normal,
                    "normal_y": vec2[0]/norm_l*sgn_normal,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=np.linalg.norm(vec2)*height,
            )
        norm_l=np.linalg.norm([vec1[1],-vec1[0]])
        side2 = SympyCurve(
                functions={
                    "x": origin[0]+ s1*vec1[0],
                    "y": origin[1]+ s1*vec1[1],
                    "z": origin[2]+ s2*height,
                    "normal_x": vec1[1]/norm_l*sgn_normal,
                    "normal_y": -vec1[0]/norm_l*sgn_normal,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=np.linalg.norm(vec1)*height,
            )
        
        norm_l=np.linalg.norm([-vec2[1],vec2[0]])
        side3 = SympyCurve(
                functions={
                    "x": origin[0]+ vec1[0]+s1*vec2[0],
                    "y": origin[1]+ vec1[1]+s1*vec2[1],
                    "z": origin[2]+ s2*height,
                    "normal_x": vec2[1]/norm_l*sgn_normal,
                    "normal_y": -vec2[0]/norm_l*sgn_normal,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=np.linalg.norm(vec2)*height,
            )
        norm_l=np.linalg.norm([vec1[1],-vec1[0]])
        side4 = SympyCurve(
                functions={
                    "x": origin[0]+ s1*vec1[0]+vec2[0],
                    "y": origin[1]+ s1*vec1[1]+vec2[1],
                    "z": origin[2]+ s2*height,
                    "normal_x": -vec1[1]/norm_l*sgn_normal,
                    "normal_y": vec1[0]/norm_l*sgn_normal,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=np.linalg.norm(vec1)*height,
            )               
        
        sides=[top,bottom,side1,side2,side3,side4]
        
        # compute SDF in 2D Parallelogram
        points = [origin[0:2],corner1[0:2],corner3[0:2] ,corner2[0:2]]
         # wrap points
        wrapted_points = points + [points[0]]
         
        sdfs = [(x - wrapted_points[0][0]) ** 2 + (y - wrapted_points[0][1]) ** 2]
        conds = []
        for v1, v2 in zip(wrapted_points[:-1], wrapted_points[1:]):
            # sdf calculation
            dx = v1[0] - v2[0]
            dy = v1[1] - v2[1]
            px = x - v2[0]
            py = y - v2[1]
            d_dot_d = dx**2 + dy**2
            p_dot_d = px * dx + py * dy
            max_min = Max(Min(p_dot_d / d_dot_d, 1.0), 0.0)
            vx = px - dx * max_min
            vy = py - dy * max_min
            sdf = vx**2 + vy**2
            sdfs.append(sdf)

            # winding calculation
            cond_1 = Heaviside(y - v2[1])
            cond_2 = Heaviside(v1[1] - y)
            cond_3 = Heaviside((dx * py) - (dy * px))
            all_cond = cond_1 * cond_2 * cond_3
            none_cond = (1.0 - cond_1) * (1.0 - cond_2) * (1.0 - cond_3)
            cond = 1.0 - 2.0 * Min(all_cond + none_cond, 1.0)
            conds.append(cond)

        # set inside outside
        sdf_xy = Min(*sdfs)
        cond = reduce(mul, conds)
        sdf_xy = sqrt(sdf_xy) * -cond
        
        
        #compute distance in z-direction in 3D         
        center_z = origin[2] + height / 2
        sdf_z = height / 2 - Abs(z - center_z)
        
        if (sdf_xy >=0) & (sdf_z >= 0):
            sdf= Min(sdf_xy,sdf_z)
        elif (sdf_xy <0)&(sdf_z>0):
            sdf = sdf_xy
        elif (sdf_xy>0)&sdf_z<0:
            sdf = sdf_z
        else: #min distance to all 12 boundary curves
            sdf = -sqrt(sdf_xy**2+sdf_z**2)
            
                
        # calculate bounds
        max_x = max(origin[0],corner1[0],corner2[0],corner3[0])
        min_x = min(origin[0],corner1[0],corner2[0],corner3[0])
        max_y = max(origin[1],corner1[1],corner2[1],corner3[1])
        min_y = min(origin[1],corner1[1],corner2[1],corner3[1])
        
        bounds = Bounds(
            {
                Parameter("x"): (min_x, max_x),
                Parameter("y"): (min_y, max_y),
                Parameter("z"): (float(origin[2]), float(origin[2]+height)),
            },
            parameterization=parameterization,
        )
        
        # initialize ParallelogramCylinder
        super().__init__(
            sides,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
