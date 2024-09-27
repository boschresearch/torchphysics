import inspect
from typing import Dict
from torch import Tensor
import torch.nn as nn
import torch
import numpy as np
import logging


class TPNodeFunction(nn.Module):
    """
    Module to evaluates a given TorchPhysics function (objective) with 
    the given input variables.
    In the forward call it converts the input variables to the format 
    of the TP objective function and evaluates it.
    It returns the result of the objective function as a dictionary 
    with the name of the objective function as key and the norm of the 
    result as value.

    Parameters
    ----------
    tpfunction : callable
        The TorchPhysics function to evaluate.
    input_space : dict
        The input space of the objective function.
    output_space : dict
        The output space of the objective function.
    data_functions : dict
        The data functions of the objective function.
    params : list
        The parameters of the objective function.    

    """ 

    def __init__(self,tpfunction,input_space,output_space,data_functions,params):    
        super().__init__()                        
        self.objective = tpfunction                
        self.input_space = input_space
        self.output_space = output_space
        self.data_functions = data_functions
        self.params = params        
        variables = set(inspect.signature(self.objective.fun).parameters)            
        input_keys, output_keys = self.getInOutputVariables()    
        parameter_variables=variables-self.input_space.keys()-self.output_space.keys()
                
        parameter_variables_new = set([])
        for param in parameter_variables:
            if param in params[0].space.variables:
                if params[0].space.dim>1:
                    parameter_variables_new = parameter_variables_new | set([param+str(l) for l in range(params[0].space.dim)])
                else:
                    parameter_variables_new = parameter_variables_new | set(param)
        self.parameter_variables=parameter_variables_new        
        self.variables = ((parameter_variables_new)|input_keys|output_keys)-set(self.data_functions.keys())
        self.cond_names = [tpfunction.fun.__name__]

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:         
        invars = self.convertVariables(in_vars)
        objectives = self.objective(invars)               
        res = {self.cond_names[0]: torch.norm(objectives,dim=1)}
        return res
  
    def getInOutputVariables(self):
        '''
        Returns the input keys, output keys and spatial keys of the 
        objective function.        
        '''
        
        spatial_keys = set([])
        inputkeys = set([])
        outputkeys = set([])
        
        if {'x','y','z'} <= set(self.input_space.keys()):
            spatial_keys = {'x','y','z'}
        elif {'x','y'} <= set(self.input_space.keys()):
            spatial_keys = {'x','y'}
        elif {'x'} <= set(self.input_space.keys()):
            if self.input_space['x'] > 1:
                if self.input_space['x'] ==2:         
                    spatial_keys = {'x','y'}
                else:                    
                    spatial_keys = {'x','y','z'}
            else:                
                spatial_keys = {'x'}
        
        for key in set(self.input_space.keys())- {'x','y','z'}:
            if self.input_space[key] >1:                
                inputkeys = {key+str(l+1) for l in range(self.input_space[key])}
            else:
                inputkeys = inputkeys|{key}
        for key in set(self.output_space.keys()):
            if self.output_space[key] >1:                
                outputkeys = outputkeys|{key+str(l+1) for l in range(self.output_space[key])}
            else:
                outputkeys = outputkeys|{key}
        
        
        return spatial_keys|inputkeys, outputkeys
    
    def convertVariables(self,vars):        
        '''
        Converts the input variables to the format of the TP 
        objective function
        '''
        conv_vars = vars.copy()
        
        if {'x','y','z'} <= set(self.input_space.keys()):
            pass            
        elif {'x','y'} <= set(self.input_space.keys()):
            pass            
        elif {'x'} <= set(self.input_space.keys()):
            if self.input_space['x'] > 1:
                if self.input_space['x'] ==2:
                    conv_vars['x']=torch.cat((vars['x'],vars['y']),dim=1)
                    conv_vars.pop('y')                
                else:
                    conv_vars['x']=torch.cat((vars['x'],vars['y'],vars['z']),dim=1)
                    conv_vars.pop('y')
                    conv_vars.pop('z')
        
        for key in set(self.input_space.keys())- {'x','y','z'}:
            if self.input_space[key] >1:
                cat_var = list(vars[key+str(l+1)] for l in range(self.input_space[key]))
                conv_vars[key] = torch.cat(cat_var,dim=1)
                for l in range(self.input_space[key]):
                    conv_vars.pop(key+str(l+1))                
        
        for key in self.output_space.keys():
            if self.output_space[key] >1:
                if self.input_space['x'] > 1:   
                    if self.input_space['x'] ==2:
                        cat_var = list(OutvarFunction.apply(vars[key+str(l+1)], conv_vars['x'],vars['x'],vars['y']) for l in range(self.output_space[key]))
                    else:
                        cat_var = list(OutvarFunction.apply(vars[key+str(l+1)], conv_vars['x'],vars['x'],vars['y'],vars['z']) for l in range(self.output_space[key]))                   
                    conv_vars[key] = torch.cat(cat_var,dim=1)                     
                else:
                    cat_var = list(vars[key+str(l+1)] for l in range(self.output_space[key]))
                    conv_vars[key] = torch.cat(cat_var,dim=1)
                    for l in range(self.output_space[key]):
                        conv_vars.pop(key+str(l+1))
            else:
                if self.input_space['x'] > 1:                    
                    if self.input_space['x'] ==2:                        
                        conv_vars[key] = OutvarFunction.apply(vars[key], conv_vars['x'],vars['x'],vars['y'])
                    else:
                        conv_vars[key] = OutvarFunction.apply(vars[key], conv_vars['x'],vars['x'],vars['y'],vars['z'])
                          
        for param in self.params[0].space.variables:
            if self.params[0].space.dim>1:
                conv_vars[param] = torch.cat(list(vars[param+str(l)] for l in range(self.params[0].space.dim)),dim=1)
                for l in range(self.params[0].space.dim):
                        conv_vars.pop(param+str(l))
            else:
                conv_vars[param] = vars[param]
                
        for fun in self.data_functions.keys():
            conv_vars[fun] = self.data_functions[fun](conv_vars)
            
        return conv_vars
        
        
class OutvarFunction(torch.autograd.Function):
    """
    Function that calculates the gradient of the output variable with 
    respect to a multi-dimensional spatial input variable, if the 
    gradient function is given for the corresponding one-dimensional 
    spatial input variable. 
    Variables x,y,z are the one-dimensional spatial input variables and
    x_vec is the multi-dimensional spatial input variable.
    """
    @staticmethod
    def forward(ctx, u, x_vec, x, y, z=None):
        """
        Forward pass of the node.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): Context 
                object for autograd.
            u (torch.Tensor): Input tensor.
            x_vec (torch.Tensor): Input tensor.
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Input tensor.
            z (torch.Tensor, optional): Input tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Compute u(u, xy) (new u) during forward pass
        result = u

        # Save u and xy for the backward pass       
        if z is not None:
            ctx.save_for_backward(u, x_vec, x, y, z)
        else:
            ctx.save_for_backward(u, x_vec, x, y)
        return result

    @staticmethod
    def backward(ctx, grad_output):
            """
            Computes the backward pass for the custom autograd function.
            
            Args:
                ctx (torch.autograd.function._ContextMethodMixin): The 
                    context object that holds the saved tensors.
                grad_output (torch.Tensor): The gradient of the output 
                    with respect to the function's output.
            
            Returns:
                tuple: A tuple containing the gradients of the input 
                    tensors with respect to the function's inputs.
            """
            grad_u =  grad_output
            
            # Load u and xy from the context
            if len(ctx.saved_tensors)==5:
                u, x_vec,x,y,z  = ctx.saved_tensors
            else:
                u, x_vec, x,y  = ctx.saved_tensors
            # Compute the gradients of u (new) with respect to u and xy        
            if len(ctx.saved_tensors)==5:            
                grad_x, grad_y, grad_z = torch.autograd.grad(u, (x,y,z), grad_output, create_graph=True)
                grad_xyz = torch.cat((grad_x,grad_y,grad_z),dim=1)
                return  grad_u, OutvarFunction.apply(grad_xyz,x_vec,x,y,z), OutvarFunction.apply(grad_x,x_vec,x,y,z), OutvarFunction.apply(grad_y,x_vec,x,y,z), OutvarFunction.apply(grad_z,x_vec,x,y,z)
            else:
                grad_x, grad_y = torch.autograd.grad(u, (x,y), grad_output, create_graph=True)
                grad_xy = torch.cat((grad_x,grad_y),dim=1)
                return  grad_u, OutvarFunction.apply(grad_xy,x_vec,x,y), OutvarFunction.apply(grad_x,x_vec,x,y), OutvarFunction.apply(grad_y,x_vec,x,y)
