from torchphysics.problem.conditions.condition import PINNCondition, DataCondition, ParameterCondition
from torchphysics.problem.domains.domainoperations.product import ProductDomain
from torchphysics.problem.domains.domainoperations.union import UnionDomain, UnionBoundaryDomain
from torchphysics.problem.domains.domainoperations.intersection import IntersectionDomain, IntersectionBoundaryDomain
from torchphysics.problem.domains.domainoperations.cut import CutDomain, CutBoundaryDomain
from torchphysics.problem.domains.domainoperations.rotate import Rotate
from torchphysics.problem.domains.domainoperations.translate import Translate
from torchphysics.models.parameter import Parameter
from torchphysics.problem.spaces import Points

from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint, PointwiseConstraint
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.loss import PointwiseLossNorm


from modulus.sym.node import Node
from modulus.sym.key import Key
from modulus.sym.models.fully_connected import FullyConnectedArch

from sympy import Symbol
from functools import partial

import warnings
import torch
import sympy

from .model import TPModelArch
from .nodes import TPNodeFunction
from .geometry import TPGeometryWrapper
from .helper import convertDataModulus2TP, convertDataTP2Modulus, PointwiseLossInfNorm, PointwiseLossMean, CustomInferencerPlotter, PINNConditionValidator, DataConditionValidator

class ModulusSolverWrapper():
    """
    Wrapper to use the solver class of Modulus with the rest 
    implemented in TorchPhysics
    
    Parameters
    ----------
        
    tp_solver: torchphysics.solver.Solver (pl.LightningModule)
        TorchPhysics solver class
    callbacks: list of torchphysics.callbacks.Callback
        List of TorchPhysics callbacks
    **kwargs:
        lambda_weighting: list of floats, integers, sympy expressions 
        or the string "sdf"
            List of lambda weightings for the conditions. If the 
            string "sdf" is used, the lambda weighting is multiplied 
            with the signed distance function of the boundary condition.
            If only one value is given, it is used for all conditions. 
            If the length of the list is not equal to the number of 
            conditions, the default value of 1.0 is used for the 
            remaining conditions.
        
    Returns
    -------
    ModulusSolverWrapper
        Wrapper class for Modulus solver class containing the Modulus 
        domain with the necessary nodes, geometries, conditions and 
        parameters
                
    """
    
    def __init__(self,tpsolver,callbacks,**kwargs):
        self.nodes = []
        self.domain = Domain()

        self.lambda_weighting_vals = [1.0]*len(tpsolver.train_conditions)
        for key, value in kwargs.items():        
            if key == 'lambda_weighting':               
                assert type(value) == list, "lambda_weighting must be a list"
                assert all((type(val) in (int,float)) or (isinstance(val, sympy.Expr)) or (val=="sdf") for val in value), "lambda_weighting must be a list of floats, integers, sympy expressions or the string ""sdf""" 
                if len(value) == 1:
                    value = value*len(tpsolver.train_conditions)
                elif len(value) != len(tpsolver.train_conditions):
                    assert False, "lambda_weighting must have the same length as the number of conditions or 1"
                self.lambda_weighting_vals = value
        
        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"
         
        self.models = []  
        num_models = 0
        self.orig_models = []
        self.Modmodels = []
        self.Geometries = [] 
        self.conds = []
        self.isBoundary = []  
        self.parameters = []    
        self.parameter_samples = []
        self.parameter_nets = [] 
        objectives = []
        exist_DataCondition = False
        is_inverse_problem = False
        not_seen_before = True

        # loop through all conditions to collect nodes out of objective functions
        for condition in tpsolver.train_conditions+tpsolver.val_conditions:
            if type(condition)!=ParameterCondition:
                model = condition.module   
                self.orig_models.append(model)             
                if type(model).__name__=='Parallel':
                    models = model.models
                else:
                    models = [model]
                for mod in models:   
                    if mod not in self.models:                 
                        self.models.append(mod)
                        Modmodel = TPModelArch(mod)
                        self.Modmodels.append(Modmodel)
                        num_models +=1
                        self.nodes.append(Modmodel.make_node("model"+str(num_models-1)))               

            if (type(condition) == PINNCondition):                   
                if condition.sampler.is_static:
                    for fn in condition.data_functions:                    
                        condition.data_functions[fn].fun = condition.data_functions[fn].fun.to(self.device)                                                        
                # construct node out of objective function
                NodeFCN=TPNodeFunction(condition.residual_fn,model.input_space,model.output_space,condition.data_functions,condition.parameter)                
                objectives.append(NodeFCN)
                if condition in tpsolver.train_conditions and condition in tpsolver.val_conditions:
                    if not_seen_before:
                        self.nodes.append(Node(NodeFCN.variables, NodeFCN.cond_names, NodeFCN))
                        not_seen_before = False
                else:
                    self.nodes.append(Node(NodeFCN.variables, NodeFCN.cond_names, NodeFCN))                 
                # if there is a learnable parameter in the condition, add a parameter net to learn this parameter as Modulus does not support additional learnable parameters in the conditions.
                # The parameter is then learned as a function of the input variables of the TP model
                if (condition.parameter!=Parameter.empty())& (condition.parameter not in self.parameters):
                    is_inverse_problem = True
                    inputkeys, _= TPModelArch(model).getVarKeys(model)                    
                    outputkeys = [Key(var+str(ind+1)) for var in condition.parameter.space.variables for ind in range(condition.parameter.space.dim)] if condition.parameter.space.dim>1 else [Key(var) for var in condition.parameter.space.variables]                    
                    # the parameter net is a fully connected network with 2 layers and 30 neurons per layer
                    # the input keys are the keys of the corresponding TP model input space
                    parameter_net =  FullyConnectedArch(input_keys=inputkeys, output_keys=outputkeys, layer_size = 30, nr_layers = 2)                                       
                    self.parameter_nets.append(parameter_net)
                    self.nodes.append(parameter_net.make_node("parameter_net"))
                    self.parameters.append(condition.parameter) 
                    points = condition.sampler.sample_points()       
                    # sort out the input keys of the TP sampler and the corresponding values to the order of TP model input keys
                    self.parameter_samples.append(points[..., list(model.input_space.keys())])
            elif type(condition) == DataCondition:
                exist_DataCondition = True      
                objectives.append(None)  
            elif type(condition) == ParameterCondition:
                objectives.append(None)  
            else:
                assert False, "Only PINNCondition, DataCondition, ParameterCondition are allowed as conditions"

        assert(exist_DataCondition if is_inverse_problem else True), "DataCondition must be present for inverse problems"
        
        # loop over train conditions to build Modulus constraints out of TorchPhysics conditions
        for condition, obj, weight in zip(tpsolver.train_conditions,objectives[0:len(tpsolver.train_conditions)],self.lambda_weighting_vals):  
            if type(condition) == PINNCondition:               
                
                # identify sampler           
                # check if static sampler
                is_static = condition.sampler.is_static              
                sampler = condition.sampler.sampler if is_static else condition.sampler
                quasi_random = False
                if type(sampler).__name__ != 'RandomUniformSampler':
                    if type(sampler).__name__ == 'LHSSampler':
                        warnings.warn("Modulus only supports RandomUniformSampler or Halton sequence. Using Halton sequence instead.")
                        quasi_random = True
                    else:
                        warnings.warn("Modulus only supports RandomUniformSampler or Halton sequence. Using RandomUniformSampler instead.")
                    
                   
                # identify different types of domains and split them into spatial, time and parameter domains
                # spatial_domain is a nested dictionary containing different domain operations and the underlying TorchPhysics domains
                spatial_domain, time_domain, parameter_domain = self.TPDomainWrapper(sampler.domain)                                   
                
                # identify parameter ranges    
                param_ranges={}
                for dom in parameter_domain:
                    if dom.dim>1:       
                        for l in range(0,dom.dim):         
                            param_ranges[Symbol(list(dom.space.variables)[0]+str(l+1))]= tuple((dom.bounding_box()[2*l].item(),dom.bounding_box()[2*l+1].item()))                    
                    else:                        
                        param_ranges[Symbol(list(dom.space.variables)[0])]= tuple(dom.bounding_box().numpy())
                
                # identify time variable ranges
                if time_domain != []:
                    assert(all(dom == time_domain[0] for dom in time_domain)), "Only single time domain allowed"
                    time_domain = time_domain[0]  
                    if time_domain.dim == 1:
                        param_ranges[Symbol("t")]=tuple(time_domain.bounding_box().numpy())    
                    elif time_domain.dim == 0:
                        param_ranges[Symbol("t")]= time_domain.bounding_box()[0].item()
                    
                # Build geometry out of spatial domain                               
                is_boundary, geometry, cond, _, _= TPGeometryWrapper(spatial_domain).getModulusGeometry()                
                self.Geometries.append(geometry)
                self.conds.append(cond)
                self.isBoundary.append(is_boundary)
                
                # determine lambda_weightings
                if weight == "sdf":
                    if is_boundary:  
                        lambda_weightings = {name : condition.weight for name in obj.cond_names}                                     
                    else:
                        lambda_weightings = {name : condition.weight*Symbol("sdf") for name in obj.cond_names}
                else:
                    lambda_weightings = {name : condition.weight*weight for name in obj.cond_names}          

                assert (condition.track_gradients == True), "track_gradients must be True for PINNCondition"
               
                # add constraints to domain
                if is_boundary:
                    constraint = PointwiseBoundaryConstraint(
                        nodes=self.nodes,                 
                        geometry=geometry,
                        outvar={name : 0.0 for name in obj.cond_names},                        
                        batch_size=len(sampler.sample_points()),              
                        parameterization=param_ranges,
                        lambda_weighting = lambda_weightings,
                        fixed_dataset = is_static,
                        criteria = cond,
                        batch_per_epoch = 1,
                        quasirandom = quasi_random,
                    )      
                            
                else:
                    constraint = PointwiseInteriorConstraint(
                        nodes=self.nodes,                   
                        geometry=geometry,                        
                        outvar={name : 0.0 for name in obj.cond_names},
                        batch_size=len(sampler.sample_points()),              
                        parameterization=param_ranges,
                        lambda_weighting = lambda_weightings,
                        fixed_dataset = is_static,
                        batch_per_epoch = 1,
                        criteria = cond,
                        quasirandom = quasi_random,                        
                    )      
                self.domain.add_constraint(constraint, condition.name)
                
            
            elif type(condition) == DataCondition:
                if condition.use_full_dataset == True:
                    batch_size = len(condition.dataloader.dataset.data_points[0])
                else:
                    batch_size = condition.dataloader.dataset.batch_size

                if condition.norm == "inf":
                    norm = PointwiseLossInfNorm()
                else:
                    norm = PointwiseLossNorm(condition.norm)
                assert (condition.root == 1), "Only root=1 is allowed for DataCondition"    

                outvar=convertDataTP2Modulus(condition.dataloader.dataset.data_points[1])
                invar=convertDataTP2Modulus(condition.dataloader.dataset.data_points[0])

                # determine lambda_weightings                
                # lambda_weightings has to be dict of numpy arrays in the same length as invar
                lambda_weightings = {name : condition.weight*weight*torch.ones(len(outvar[list(outvar.keys())[0]]),1) for name in outvar.keys()} 
                
                data_constraint = PointwiseConstraint.from_numpy(
                        nodes=self.nodes,
                        invar=invar,
                        outvar=outvar,
                        loss = norm,
                        batch_size=batch_size,   
                        lambda_weighting = lambda_weightings,    
                    )
                
                # define new forward function for data constraint to include the constrain_fn
                if condition.constrain_fn:
                    def create_new_forward(condition):
                        def new_forward(self):
                            self._output_vars = self.model(self._input_vars)                                                      
                            inputvars = convertDataModulus2TP(self._input_vars,condition.module.input_space)
                            outputvars = convertDataModulus2TP(self._output_vars,condition.module.output_space)
                            constraint_output = condition.constrain_fn({**outputvars, **inputvars})                                                        
                            output_dict = {key: constraint_output[:,index:index+condition.module.output_space[key]] for index, key in enumerate(outputvars.keys())}                        
                            #output_dict = {key: constraint_output[:,index:index+condition.module.output_space[key]] for index, key in enumerate(condition.module.output_space.keys())}                                                    
                            output_vars_points = Points.from_coordinates(output_dict)                                                   
                            self._output_vars = convertDataTP2Modulus(output_vars_points)  
                            
                        return new_forward
                                    
                    data_constraint.forward = create_new_forward(condition).__get__(data_constraint, PointwiseConstraint)

                self.domain.add_constraint(data_constraint, condition.name)

            elif type(condition) == ParameterCondition:                
                assert (condition.parameter.space.dim ==1), "Only single parameter allowed for ParameterCondition"
                param_index = self.parameters.index(condition.parameter)
                
                points = self.parameter_samples[param_index]               
                modpoints = dict(zip([str(key) for key in self.parameter_nets[param_index].input_keys],[points.as_tensor[:,index].reshape(len(points),1) for index in range(len(self.parameter_nets[param_index].input_keys))]))   

                # determine lambda_weightings
                # lambda_weightings has to be dict of numpy arrays with the same length as invar               
                lambda_weightings = {condition.parameter.variables.pop() : condition.weight*weight*torch.ones(len(points),1)}  
               
                parameter_constraint = PointwiseConstraint.from_numpy(
                        nodes=self.nodes,
                        invar=modpoints,
                        outvar= {condition.parameter.variables.pop():torch.zeros(len(points),1)},                                                 
                        batch_size=batch_size,   
                        loss = PointwiseLossMean(),
                        lambda_weighting = lambda_weightings,  
                    )
                
                # create new forward function for parameter constraint to include the penalty function
                def create_new_forward_paramCond(condition):
                    def new_forward(self):
                        self._output_vars = self.model(self._input_vars)                            
                        penalty_output = condition.penalty({**self._output_vars})                     
                        self._output_vars = {condition.parameter.variables.pop(): penalty_output}                                                                        
                    return new_forward
                                               
                parameter_constraint.forward =  create_new_forward_paramCond(condition).__get__(parameter_constraint, PointwiseConstraint)                
                self.domain.add_constraint(parameter_constraint, condition.name)   

            else:
                assert False, "Condition type not yet supported! Only PINNCondition, DataCondition or ParameterCondition!"

        # loop over validation conditions to build Modulus constraints out of TorchPhysics conditions
        for condition, obj in zip(tpsolver.val_conditions,objectives[len(tpsolver.train_conditions):]):        
            if type(condition) == PINNCondition:   
                # convert sample points to Modulus format                
                samples=convertDataTP2Modulus(sampler.sample_points())
                
                # build validator
                validator = PINNConditionValidator(                        
                        nodes=self.nodes,                 
                        invar = samples,
                        output_names = obj.cond_names,                        
                        batch_size=len(samples),                                                                                                       
                        requires_grad = condition.track_gradients,
                    )      
                       
                self.domain.add_validator(validator, condition.name)                
            
            elif type(condition) == DataCondition: 
                batch_size = condition.dataloader.dataset.batch_size
                                   
                outvar=convertDataTP2Modulus(condition.dataloader.dataset.data_points[1])
                invar=convertDataTP2Modulus(condition.dataloader.dataset.data_points[0])

                validator = DataConditionValidator(                
                        nodes=self.nodes,
                        invar=invar,
                        true_outvar=outvar,                        
                        batch_size=batch_size,                           
                        requires_grad = condition.track_gradients,   
                        norm = condition.norm,
                        root = condition.root                  
                    )
                
                # define new forward function for data validator to include the constrain_fn
                if condition.constrain_fn:
                    def create_new_forward(condition):
                        def new_forward(self,invar):
                            with torch.set_grad_enabled(condition.track_gradients):                              
                                pred_outvar = self.model(invar)                                                          
                                inputvars = convertDataModulus2TP(invar,condition.module.input_space)
                                outputvars = convertDataModulus2TP(pred_outvar,condition.module.output_space)
                                constraint_output = condition.constrain_fn({**outputvars, **inputvars})
                                output_dict = {key: constraint_output[:,index:index+condition.module.output_space[key]] for index, key in enumerate(condition.module.output_space.keys())}                        
                                output_vars_points = Points.from_coordinates(output_dict)                                  
                            return convertDataTP2Modulus(output_vars_points)                                         
                        return new_forward                 
                    
                    validator.forward = create_new_forward(condition).__get__(validator, PointwiseValidator)
                self.domain.add_validator(validator, condition.name)

        # if inverse problem with single parameters to identify, add parameter monitor for each parameter
        # A parameter is learned by a parameter net, and the mean of the net output will be the parameter value and is monitored
        if is_inverse_problem:
            for index, param_obj in enumerate(zip(self.parameter_samples,self.parameter_nets)):
                points = param_obj[0]                             
                # ordered values get the corresponding Modulus input keys (if some of TP input keys are of higher dimension than 1, they are split into several Modulus input keys)      
                modpoints = dict(zip([str(key) for key in param_obj[1].input_keys],[points.as_tensor[:,index].reshape(len(points),1) for index in range(len(param_obj[1].input_keys))]))   
                
                # define mean function to compute mean of parameter net over a fixed set of input points 
                def mean_func(var, key):
                    return torch.mean(var[str(key)], dim=0)

                metrics = {"mean_"+str(key): partial(mean_func, key=key) for key in param_obj[1].output_keys}
               

                parameter_monitor = PointwiseMonitor(
                    nodes=self.nodes,
                    invar=modpoints,
                    output_names=[str(key) for key in param_obj[1].output_keys],                   
                    metrics = metrics,                    
                )
                self.domain.add_monitor(parameter_monitor, "parameter_monitor"+str(index))

                
        # if plotter callback is present, an inferencer with plotter is added to the domain
        if callbacks:            
            for callback in callbacks:                
                invars = { key: value.cpu().detach().numpy() for key, value in convertDataTP2Modulus(callback.point_sampler.sample_points()).items()}                
                callback.point_sampler.created_points = None
                
                plotter_inferencer = PointwiseInferencer(
                    nodes=self.nodes,
                    invar=invars,
                    output_names= [var+str(ind+1) if callback.model.output_space[var] >1 else var for var in callback.model.output_space.variables for ind in range(callback.model.output_space[var]) ], 
                    batch_size=len(invars),
                    plotter=CustomInferencerPlotter(callback),
                    )
            self.domain.add_inferencer(plotter_inferencer, callback.log_name)
                
                
    def TPDomainWrapper(self,domain):
        """
        Function that parses domain and splits it into spatial, time 
        and parameter domains.
        The spatial domain is recursively split into nested dictionary 
        containing the following keys:
        -'u': union
        -'i': intersection
        -'p': product
        -'c': cut
        -'r': rotate
        -'t': translate
        -'d': final domain
        The values are the partial/splitted domains
        
        It returns the spatial_domain, time_domain and parameter_domain.
        
        Parameters
        ----------
        domain: torchphysics.problem.domains.domain.Domain
            TorchPhysics domain 
        
        Returns
        -------
        spatial_domain: dict
            Nested dictionary containing the splitted domain
        time_domain: list
            List of time domains
        parameter_domain: list
            List of parameter domains

        """

        # we have to define these global variables due to the recursive nature of the function
        global parameter_domain, time_domain
        parameter_domain = []
        time_domain = []
        spatial_domain = self.splitDomains(domain)      
        return spatial_domain, time_domain, parameter_domain
    
        
    def splitDomains(self, domain):
        """
        Recursive function that splits the spatial domain into nested 
        dictionary containing the following keys:
        -'u': union
        -'i': intersection
        -'p': product
        -'c': cut
        -'r': rotate
        -'t': translate
        -'d': final domain
        The values are then the partial/splitted domains
        """
        global parameter_domain, time_domain
        if self.is_splittable(domain):
            if not hasattr(domain,'domain_a'):
                domain.domain_a = domain.domain.domain_a
                domain.domain_b = domain.domain.domain_b
        
            if self.is_xspace(domain.domain_a):        
                if self.is_xspace(domain.domain_b):
                    if type(domain) == ProductDomain:                          
                        return {'p': (self.splitDomains(domain.domain_a),self.splitDomains(domain.domain_b))}
                    elif type(domain) == UnionDomain:
                        return {'u': (self.splitDomains(domain.domain_a),self.splitDomains(domain.domain_b))}
                    elif type(domain) == UnionBoundaryDomain:                    
                        return {'ub': (self.splitDomains(domain.domain.domain_a),self.splitDomains(domain.domain.domain_b))}
                    elif type(domain) == IntersectionDomain:
                        return {'i': (self.splitDomains(domain.domain_a),self.splitDomains(domain.domain_b))}
                    elif type(domain) == IntersectionBoundaryDomain:
                        return {'ib': (self.splitDomains(domain.domain.domain_a),self.splitDomains(domain.domain.domain_b))}
                    elif type(domain) == CutDomain:
                        return {'c': (self.splitDomains(domain.domain_a),self.splitDomains(domain.domain_b))}
                    elif type(domain) == CutBoundaryDomain:
                        return {'cb': (self.splitDomains(domain.domain.domain_a),self.splitDomains(domain.domain.domain_b))}
                else:
                    if self.is_tspace(domain.domain_b):                        
                        time_domain.append(domain.domain_b)                
                    else:
                        parameter_domain.append(domain.domain_b)                
                    return self.splitDomains(domain.domain_a)

            else:
                if self.is_xspace(domain.domain_b):
                    if self.is_tspace(domain.domain_a):
                        time_domain.append(domain.domain_a)                
                    else:
                        parameter_domain.append(domain.domain_a)
                    return self.splitDomains(domain.domain_b)
                else:
                    if self.is_tspace(domain.domain_a):
                        time_domain.append(domain.domain_a)                
                    else:
                        parameter_domain.append(domain.domain_a)
                    if self.is_tspace(domain.domain_b):
                        time_domain.append(domain.domain_b)                
                    else:
                        parameter_domain.append(domain.domain_b)
                    return None                
        
        elif type(domain) == Rotate:
            # in the case of rotation, we have to collect the rotation function and the rotation center for later use
            return {'r': [self.splitDomains(domain.domain),domain.rotation_fn(),domain.rotate_around()]}
        elif type(domain) == Translate:
            # in the case of translation, we have to collect the translation function for later use
            return {'t': [self.splitDomains(domain.domain),domain.translate_fn()]}    
        else:
            if self.is_xspace(domain):                
                return {'d': domain}
            else:
                if self.is_tspace(domain):
                    time_domain.append(domain)                
                else:
                    parameter_domain.append(domain)
                return None
   
    def is_xspace(self,domain):    
        return ('x' in domain.space.variables)|('y' in domain.space.variables)|('z' in domain.space.variables)

    def is_tspace(self,domain):    
        return 't' in domain.space.variables

    def is_splittable(self,domain):
        return type(domain) in set((ProductDomain,UnionDomain,UnionBoundaryDomain,IntersectionDomain,IntersectionBoundaryDomain,CutBoundaryDomain,CutDomain))



