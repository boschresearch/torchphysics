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

import torch
import numpy as np

from torchphysics.problem.spaces.points import Points
from torchphysics.utils.plotting.plot_functions import plot
from modulus.sym.loss import Loss
from modulus.sym.utils.io import InferencerPlotter

from modulus.sym.domain.validator import Validator
from modulus.sym.node import Node

from modulus.sym.dataset import DictInferencePointwiseDataset, DictPointwiseDataset

from modulus.sym.domain.constraint import Constraint
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.constants import TF_SUMMARY
from modulus.sym.distributed import DistributedManager
from modulus.sym.utils.io.vtk import var_to_polyvtk

from typing import List, Dict
from typing import Dict
from torch import Tensor

def convertDataTP2Modulus(points):
    """
    Convert data from TorchPhysics to Modulus format    

    Parameters
    ----------
        
    data: Points (torchphysics.problem.space.Points)
        Dictionary containing the data points of the TorchPhysics 
        dataset with the TorchPhysics variable names as keys and the 
        corresponding data points as values.          
        
    """
    data=points.coordinates
    
    outdata={}
        
    for var in points.space.variables:                                
            if var =='x':
                outdata['x'] = data[var][:,0].unsqueeze(1)
                if points.space[var] > 1:                                                
                    outdata['y'] = data[var][:,1].unsqueeze(1)
                if points.space[var]== 3:            
                    outdata['z'] = data[var][:,2].unsqueeze(1)

            else:        
                if points.space[var]>1:
                    for ind in range(points.space[var]):
                        outdata[var+str(ind+1)]=data[var][:,ind].unsqueeze(1)
                else:
                    outdata[var]=data[var]
                        
    return outdata


def convertDataModulus2TP(data,TP_space):
    """
    Convert data from Modulus to TorchPhysics format. If a key of 
    TP_space is not present in data, it will be ignored.

    Parameters
    ----------
        
    data: dict[str,torch.Tensor]
        Dictionary containing the data points of the Modulus dataset 
        with the Modulus variable names as keys and the corresponding 
        data points as values    

    """
    outdata={}
   
    
    for key in TP_space.variables:  
        if TP_space[key] > 1:
            if key !='x':
                if all([key+str(l+1) in data.keys() for l in range(TP_space[key])]):
                    cat_var = list(data[key+str(l+1)] for l in range(TP_space[key]))
                    outdata[key] = torch.cat(cat_var,dim=1)                
            else:            
                if TP_space[key] ==2:
                    if ('x' in data.keys()) and ('y' in data.keys()):
                        outdata['x']=torch.cat((data['x'],data['y']),dim=1)
                else:
                    if ('x' in data.keys()) and ('y' in data.keys()) and ('z' in data.keys()):
                        outdata['x']=torch.cat((data['x'],data['y'],data['z']),dim=1)                
                    
        else:
            if key in data.keys():                        
                outdata[key]=data[key]   
            

    return outdata
    

class PointwiseLossInfNorm(Loss):
    """
    L-inf loss function for pointwise data
    Computes the maximum norm loss of each output tensor
    """

    def __init__(self):
        super().__init__()
        

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,        
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            )
            if "area" in invar.keys():
                l *= invar["area"]
            losses[key] = torch.max(l)
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseLossInfNorm._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step
        )    

class PointwiseLossMean(Loss):
    """
    Computes the mean of loss function values    
   
    """

    def __init__(self):
        super().__init__()
        

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,        
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            losses[key]= torch.mean(lambda_weighting[key] * torch.abs(pred_outvar[key] - true_outvar[key]))
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseLossMean._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step
        )    




def OptimizerNameMapper(tp_name):
    """
    Maps the optimizer name to intern Modulus names. If the optimizer 
    is not defined, it returns 'not defined'.
    """
    if tp_name == 'Adam':
        return 'adam'
    elif tp_name == 'LBFGS':
        return 'bfgs'
    elif tp_name == 'SGD':
        return 'sgd'
    elif tp_name == 'Adahessian':
        return 'adahessian'
    elif tp_name == 'Adadelta':
        return 'adadelta'
    elif tp_name == 'Adagrad':
        return 'adagrad'
    elif tp_name == 'AdamW':
        return 'adamw'
    elif tp_name == 'SparseAdam':
        return 'sparse_adam'
    elif tp_name == 'Adamax':
        return 'adamax'
    elif tp_name == 'ASGD':
        return 'asgd'
    elif tp_name == 'NAdam':
        return 'Nadam'
    elif tp_name == 'RAdam':
        return 'Radam'
    elif tp_name == 'RMSprop':
        return 'rmsprop'
    elif tp_name == 'Rprop':
        return 'rprop'
    elif tp_name == 'A2GradExp':
        return 'a2grad_exp'
    elif tp_name == 'A2GradInc':
        return 'a2grad_inc'
    elif tp_name == 'A2GradUni':
        return 'a2grad_uni'
    elif tp_name == 'AccSGD':
        return 'accsgd'
    elif tp_name == 'AdaBelief':
        return 'adabelief'
    elif tp_name == 'AdaBound':
        return 'adabound'
    elif tp_name == 'AdaMod':
        return 'adamod'
    elif tp_name == 'AdaFactor':
        return 'adafactor'
    elif tp_name == 'AdamP':
        return 'adamp'
    elif tp_name == 'AggMo':
        return 'aggmo'
    elif tp_name == 'Apollo':
        return 'apollo'
    elif tp_name == 'DiffGrad':
        return 'diffgrad'
    elif tp_name == 'Lamb':
        return 'lamb'
    elif tp_name == 'NovoGrad':
        return 'novograd'
    elif tp_name == 'PID':
        return 'pid'
    elif tp_name == 'QHAdam':
        return 'qhadam'
    elif tp_name == 'MADGRAD':
        return 'madgrad'
    elif tp_name == 'QHM':
        return 'qhm'
    elif tp_name == 'Ranger':
        return 'ranger'
    elif tp_name == 'RangerQH':
        return 'ranger_qh'
    elif tp_name == 'RangerVA':
        return 'ranger_va'
    elif tp_name == 'SGDW':
        return 'sgdw'         
    elif tp_name == 'SGDP':
        return 'sgdp'
    elif tp_name == 'SWATS':
        return 'swats'
    elif tp_name == 'Shampoo':
        return 'shampoo'
    elif tp_name == 'Yogi':
        return 'yogi'
    else:
        return 'not defined'

def SchedulerNameMapper(tp_name):  
    """
    Maps the scheduler name to intern Modulus names. If the scheduler 
    is not defined, it returns 'not defined'.
    """

    if tp_name == 'ExponentialLR':
        return 'exponential_lr'
    elif tp_name == 'TFExponentialLR':
        return 'tf_exponential_lr'
    elif tp_name == 'CosineAnnealingLR':
        return 'cosine_annealing'
    elif tp_name == 'CosineAnnealingWarmRestarts':
        return 'cosine_annealing_warm_restarts'
    else:
        return 'not defined'

def AggregatorNameMapper(tp_name):
    """
    Maps the aggregator name to intern Modulus names. If the aggregator 
    is not defined, it returns 'not defined'.
    """
    if tp_name == 'Sum' :
        return 'sum'
    elif tp_name == 'GradNorm':
        return 'grad_norm'
    elif tp_name == 'ResNorm':
        return 'res_norm'
    elif tp_name == 'Homoscedastic':
        return 'homoscedastic'
    elif tp_name == 'LRAnnealing':
        return 'lr_annealing'
    elif tp_name == 'SoftAdapt':
        return 'soft_adapt' 
    elif tp_name == 'Relobralo':
        return 'relobralo'
    else:
        return 'not defined'
    
class CustomInferencerPlotter(InferencerPlotter):
    """
    Custom inferencer plotter class for using TorchPhysics plot 
    callbacks
    """
    def __init__(self,callback):
        self.plot_function = callback.plot_function
        self.plot_type = callback.plot_type
        self.angle = callback.angle
        self.point_sampler = callback.point_sampler
        self.kwargs = callback.kwargs
        self.model = callback.model

    def __call__(self, invar, outvar):        
        fig = plot(model=self.model, plot_function=self.plot_function,
                       point_sampler=self.point_sampler, 
                       angle=self.angle, plot_type=self.plot_type,
                       device=next(self.model.parameters()).device, **self.kwargs)
            
        return [(fig,'')]

       


class PINNConditionValidator(Validator):
    """
    Pointwise Validator that allows validating output variables of the 
    Graph on pointwise data.

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    invar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays as input.
    output_names : List[str]
        List of desired outputs.
    batch_size : int, optional
            Batch size used when running validation, by default 1024    
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        output_names: List[str],
        batch_size: int = 1024,        
        requires_grad: bool = False,        
    ):

         # get dataset and dataloader
        self.dataset = DictInferencePointwiseDataset(
            invar=invar, output_names=output_names
        )
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            distributed=False,
            infinite=False,
        )      
        

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )

        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set forward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = None
    

    
    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        #true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0,) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        # TODO add metrics specific for validation
        # TODO: add potential support for lambda_weighting
        #losses = PointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)
        losses = {key: torch.mean(value) for key, value in pred_outvar_cpu.items()}

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}        
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:
            var_to_polyvtk(
                {**invar,  **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,               
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name , loss, step, new_style=True
                )
        return losses        
    

class DataConditionValidator(Validator):
    """
    Validator that allows validating on pointwise data.
    The validation error is the cumulative norm over all output 
    variables. The norm can be defined by the user.

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    invar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays as input.
    true_outvar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays used to validate against validation.
    batch_size : int, optional
        Batch size used when running validation, by default 1024    
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    norm:  int or 'inf', optional
        The 'norm' which should be computed for evaluation. If 'inf', 
        maximum norm will be used. Else, the result will be taken to 
        the n-th potency (without computing the root!)
    root: int, optional
        The root of the norm. If norm is 'inf', this parameter will be 
        ignored.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int = 1024,       
        requires_grad: bool = False,
        norm: int = 2,
        root: int = 1,
    ):

        # get dataset and dataloader
        self.dataset = DictPointwiseDataset(invar=invar, outvar=true_outvar)
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            distributed=False,
            infinite=False,
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set forward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = None

        self.norm = norm
        self.root = root

    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        # TODO add metrics specific for validation
        # TODO: add potential support for lambda_weighting
        #losses = PointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)
        loss=torch.zeros(1)
        for key in true_outvar_cpu.keys():
            if self.norm == 'inf':
                loss = torch.max(loss,torch.max(torch.abs(true_outvar_cpu[key] - pred_outvar_cpu[key])))
            else:            
                loss += torch.mean(torch.abs(true_outvar_cpu[key] - pred_outvar_cpu[key])**self.norm)
        
        losses ={'': loss**1/self.root}
        
        

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:
            var_to_polyvtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name, loss, step, new_style=True
                )
        return losses
    