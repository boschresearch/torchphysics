from modulus.sym.hydra.utils import compose
from modulus.sym.solver import Solver

from torchphysics.models import Parameter
import torch

import os
import csv
import numpy as np

from .helper import OptimizerNameMapper, SchedulerNameMapper, AggregatorNameMapper
from .solver import ModulusSolverWrapper

import shutil
import warnings
import logging
# Set the logging level for matplotlib to WARNING
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)


class TPModulusWrapper():   
    '''
    Training of a TorchPhysics trainer/solver with the Modulus wrapper.
    The wrapper is a bridge between TorchPhysics and Modulus. It uses
    the Modulus configuration and the Modulus solver to train the 
    TorchPhysics solver/trainer/models.
    Loss weighting algorithms can be selected by choosing an
    aggregation function. The aggregation function can be selected by
    the parameter "aggregator" and additional arguments can be set by 
    the parameter "aggregator_args".
    A learning rate scheduler can be selected by the parameter 
    "scheduler" and additional arguments can be set by the parameter 
    "scheduler_args".
    Pointwise weighting of the loss can be set by the parameter 
    "lambda_weighting". The pointwise weighting can be a list of 
    numbers or sympy expressions or the string 'sdf'.        
    Notes        
    ----- 
    The following conventions are important for the usage of the 
    wrapper:
        Possible spatial variables: x, y, z or x (multidimensional)
        Time variable: t
        Geometries: TP geometries (domains) should be defined in the 
                    space of x, y, z or x (multidimensional).
                    A general product of domains as in TorchPhysics 
                    can not be implemented in Modulus, because the 
                    domain defines a spatial geometry that must have a
                    sdf implementation which is not available in 
                    general for an arbitrary product domain.
                    Cross products of domains and domain operations are
                    generally allowed, but too complicated 
                    constructions should be avoided, e.g. a cross 
                    product of 3 translated intervals is not allowed or
                    only isosceles triangle with axis of symmetry 
                    parallel to y-axis in cross product with an interval 
                    are supported.
                    Shapely polygons in 2D are supported, but currently
                    3D geometries (TrimeshPolyhedron) defined in stl-
                    files are only supported in Modulus in the container 
                    installation.
                    Translation of primitive domains is supported, but 
                    not translation of domains resulting from domain 
                    operations like union, intersection, difference.
                        
    Parameters
    ----------        
    trainer : pytorch_lightning.Trainer
        The Pytorch Lightning Trainer instance. 
        Supported parameters of trainer instance:
            Modulus always uses GPU device if available. Trainer 
            settings concerning GPU devices or cuda handling, e.g. 
            'accelerator' or 'devices', are not supported by this
            wrapper.
            Modulus automatically logs the training process with 
            tensorboard. The tensorboard logs are saved in the output 
            directory.
            All TorchPhysics callbacks are supported by the wrapper.
            The following Trainer parameters are supported by this 
            wrapper:
                'max_steps' : int, optional
                    The maximum number of training steps. If not 
                    specified, the default value of Pytorch Lightning 
                    Trainer is used.
                'val_check_interval' : int optional
                    How often to check the validation set. Default is 
                    1.0, meaning once per training epoch.
                'log_every_n_steps' : int, optional
                    How often to log within steps. Modulus/wrapper 
                    default is 50.                    
            Checkpoints, progress bar and model summary are 
            automatically used by Modulus.

    solver: torchphysics.solvers.Solver
        The TorchPhysics solver instance.
        All parameters of the TorchPhysics solver are supported by the 
        wrapper.
    outputdir_name : str, optional
        The name of the Modulus output directory, where the trained 
        models, the optimization configuration, tensorboard files, etc. 
        are saved. Default is 'outputs'.
        If the directory contains the results of a previous run and the
        configuration of a second call is mainly changed, there will be 
        a conflict loading existing models or configuration leading to 
        an error.
        If the directory contains the results of a previous run and the
        configuration of a second call is mainly unchanged, the new run 
        will continue the previous run with the already trained Modulus 
        models.
        If not desired or in error case, it is recommended to remove 
        the content of the output directory before starting a new run.
    confdir_name : str, optional 
        The name of a Modulus configuration directory, where initially 
        a hydra configuration file is saved. It is overwritten on each 
        call.  Default is 'conf'.        
    keep_output : bool, optional. Default is True.
        If True, the output directory is not deleted after the training
        process. Otherwise, it is deleted after the training process.    
    **kwargs : optional
        Additional keyword arguments:
            "lambda_weighting": list[Union[int, float, sp.Basic]]=None                    
                The spatial pointwise weighting of the constraint. It 
                is a list of numbers or sympy expressions or the string
                'sdf'. 
                If the list has more than one element, the length of 
                the list and the order has to match the number of 
                TorchPhysics conditions in the call 
                tp.solver.Solver([condition_1,condition_2,...]).
                If the list has only one element, the same weighting is
                applied to all conditions.
                If it is a sympy expression, it has to be a function of 
                the spatial coordinates x, y, z.
                If the TorchPhysics conditions contain weight 
                definitions with the keyword "weight", these are 
                additionally applied.
                For example,
                'lambda_weighting=["sdf"]' would apply a pointwise 
                weighting of the loss by the signed distance function, 
                but only for interior sampling, not boundary sampling.
                'lambda_weighting=[100.0, 2.0] would apply a pointwise 
                weighting of the loss by 100 to the first TorchPhysics 
                condition and 2 to the second TorchPhysics condition.
                'lambda_weighting=[2.0*sympy.Symbol('x')]' would apply 
                a pointwise weighting to the loss of `2.0 * x`.
            "aggregator" : str = None
                The aggregation function for the loss. It is a string 
                with the name of the aggregation function. Default is 
                'Sum'.
                Possible values are 'Sum', 'GradNorm', 'ResNorm, 
                'Homoscedastic','LRAnnealing','SoftAdapt','Relobralo'.
            "aggregator_args" : dict = None
                Additional arguments for the aggregation function. It 
                is a dictionary with the argument names as keys and the
                argument values as values. Default is None.
                Possible arguments with its default values are, 
                depending on the aggregator:
                GradNorm: 
                    alpha = 1.0
                ResNorm:
                    alpha = 1.0
                LRAnnealing:
                    update_freq = 1
                    alpha = 0.01
                    ref_key = None  # Change to Union[None, str] when 
                    supported by hydra
                    eps = 1e-8
                SoftAdapt:
                    eps = 1e-8
                Relobralo:
                    alpha = 0.95
                    beta = 0.99
                    tau = 1.0
                    eps = 1e-8
            "scheduler" : str = None
                The learning rate scheduler. It is a string with the 
                name of the scheduler. Default is constant learning 
                rate.    
                Possible values are 'ExponentialLR', 
                'CosineAnnealingLR' or 'CosineAnnealingWarmRestarts'.          
            "scheduler_args" : dict = None
                Additional arguments for the scheduler. It is a 
                dictionary with the argument names as keys and the 
                argument values as values. Default is None.
                Possible arguments with its default values are, 
                depending on the scheduler:
                ExponentialLR:
                    gamma = 0.99998718
                TFExponentialLR:
                    decay_rate = 0.95
                    decay_steps = 1000
                CosineAnnealingLR:
                    T_max = 1000
                    eta_min = 0.0
                    last_epoch= -1
                CosineAnnealingWarmRestarts:
                    T_0 = 1000
                    T_mult = 1
                    eta_min = 0.0
                    last_epoch = -1
    ''' 
    def __init__(self,trainer,solver,outputdir_name = 'outputs',confdir_name = 'conf',keep_output = True, **kwargs):     
        self.outputdir_name = outputdir_name
        self.keep_output = keep_output

        self.logger=logging.getLogger()
        self.ch = logging.StreamHandler()
        self.logger.addHandler(self.ch)

        # get the absolute path of the conf-directory
        caller_path=os.path.abspath(os.getcwd()+'/'+confdir_name)
        
        # Get the relative path of the current file to the conf-directory
        current_path = os.path.relpath(caller_path,os.path.dirname(os.path.abspath(__file__)))        

        if 'aggregator' in kwargs.keys():                    
            aggregator_name = AggregatorNameMapper(kwargs['aggregator'])
            assert (aggregator_name != 'not defined'), "This aggregator class is currently not supported by Modulus!"
        else:
            aggregator_name = AggregatorNameMapper('Sum')

        optimizer_name = OptimizerNameMapper(solver.optimizer_setting.optimizer_class .__name__)
        assert (optimizer_name != 'not defined'), "This optimizer class is currently not supported by Modulus!"
        assert ((solver.optimizer_setting.scheduler_class is None) or (kwargs.get('scheduler') is None)), "The scheduler should either be defined in the optimizer settings or as additional parameter of the TPModulusWrapper!"
        if solver.optimizer_setting.scheduler_class is not None:            
            scheduler_name = SchedulerNameMapper(solver.optimizer_setting.scheduler_class.__name__)
        elif kwargs.get('scheduler') is not None:
            scheduler_name = SchedulerNameMapper(kwargs.get('scheduler'))
        else:
            scheduler_name = 'exponential_lr'

        assert (scheduler_name != 'not defined'), "This scheduler class is currently not supported by Modulus!"
        
        os.makedirs(caller_path, exist_ok=True)        
        with open(caller_path+'/config_Modulus.yaml', 'w') as f:
            f.write('defaults :\n  - modulus_default\n  - loss: '+aggregator_name+'\n  - optimizer: '+optimizer_name+'\n  - scheduler: '+scheduler_name+'\n  - _self_\n')       
        self.cfg = compose(config_path=current_path, config_name="config_Modulus")
       
        training_rec_results_freq = self.cfg.training.rec_results_freq
        
        # as the initialization without scheduler leads to an error and additionally the constant LR scheduler is not implemented as class,
        # we use the exponential LR scheduler with gamma=1 for constant lr        
        if (solver.optimizer_setting.scheduler_class is None) and (kwargs.get('scheduler') is None):
            self.cfg.scheduler.gamma = 1.0
        else:             
            if solver.optimizer_setting.scheduler_args is not None:
                for key, value in solver.optimizer_setting.scheduler_args.items():
                    self.cfg.scheduler[key]=value
            if kwargs.get('scheduler_args') is not None:
                for key, value in kwargs.get('scheduler_args').items():
                    self.cfg.scheduler[key]=value   
                  
        assert (solver.optimizer_setting.scheduler_frequency == 1), "The scheduler frequency is not supported in Modulus!"
        
        for key, value in solver.optimizer_setting.optimizer_args.items():
            self.cfg.optimizer[key]=value
        
        self.cfg.network_dir = self.outputdir_name   
        
        self.cfg.training.max_steps = trainer.max_steps
        self.cfg.optimizer.lr = solver.optimizer_setting.lr

        
        self.cfg.training.rec_results_freq = min(training_rec_results_freq,trainer.max_steps)
        
        if (type(trainer.val_check_interval)) is float:           
            self.cfg.training.rec_validation_freq =int(trainer.val_check_interval*self.cfg.training.max_steps)
        else:
            self.cfg.training.rec_validation_freq = trainer.val_check_interval
        self.cfg.training.summary_freq = trainer.log_every_n_steps
        
        self.weight_save_callback = None
        self.checkpoint_callback = None
        callbacks2Modulus=[]
      
        for callback in trainer.callbacks:
            if type(callback).__name__=='WeightSaveCallback':
                self.weight_save_callback = callback
                if self.weight_save_callback.check_interval >0:
                    warnings.warn('The option check_interval of the WeightSaveCallback with check for minimial loss is not supported by Modulus. Only initial and final model state saves.')
            elif type(callback).__name__=='PlotterCallback':
                self.cfg.training.rec_inference_freq = callback.check_interval
                callbacks2Modulus.append(callback)
            elif type(callback).__name__=='TrainerStateCheckpoint':                
                self.checkpoint_callback = callback
                self.cfg.training.save_network_freq  = self.checkpoint_callback.check_interval
                warnings.warn('TorchPhysics TrainerStateCheckpoint callback is requested. The checkpointing will be automatically done by Modulus and training can be restarted. The option weights_only is not supported by Modulus. Please use the WeightSaveCallback instead.')
        
                 
                    
        self.Msolver=ModulusSolverWrapper(solver,callbacks2Modulus,**kwargs)
                
        # adapt cfg-parameters
        for key, value in kwargs.items():
            if key == 'aggregator_args':
                for key2, value2 in value.items():
                    self.cfg.loss[key2] = value2
        

         # Modulus solver instance is created
        self.slv = Solver(self.cfg, self.Msolver.domain)


    def train(self,resume_from_ckpt=False):
        '''
        Call the training process of the Modulus solver. The training 
        process is started with the Modulus configuration and the 
        Modulus solver.
        The TorchPhysics models are trained and the function 
        additionally returns trained parameters.

        Parameters
        ----------
        resume_from_ckpt: bool, optional. Default is False.
        If True, the training is resumed from the Modulus checkpoint files.
        
        '''
        # if a TorchPhysics checkpoint callback exists and the training is started without resume option, delete Modulus checkpoint and model files in the Modulus output directory.
        if not resume_from_ckpt:               
            if os.path.isdir(os.path.abspath(os.getcwd()+'/'+self.outputdir_name)):
                    shutil.rmtree(os.path.abspath(os.getcwd()+'/'+self.outputdir_name))     

        # save initial model before training              
        if self.weight_save_callback:
            if self.weight_save_callback.save_initial_model:
                torch.save(self.weight_save_callback.model.state_dict(), self.weight_save_callback.path+'/' + self.weight_save_callback.name + '_init.pt')
        # start training                
        self.slv.solve()

        
        for model in self.Msolver.models:
            model.to('cpu')
        for model in self.Msolver.orig_models:
            model.to('cpu')
           
        result_vec=[]
        for index, param_obj in enumerate(zip(self.Msolver.parameters,self.Msolver.parameter_nets)):            
            for outvar in param_obj[1].output_keys:                    
                with open(os.path.abspath(os.getcwd()+'/'+self.outputdir_name+'/monitors/mean_'+str(outvar)+'.csv'), 'r') as f:
                    reader = csv.reader(f)
                    key_line = next(reader)
                    last_line = list(reader)[-1]                        
                    result_vec.append(float(last_line[1]))
                
            self.Msolver.parameters[index]._t=torch.tensor([result_vec])
        if self.weight_save_callback:
            if self.weight_save_callback.save_final_model:
                torch.save(self.weight_save_callback.model.state_dict(), self.weight_save_callback.path+'/' + self.weight_save_callback.name + '_final.pt')
        
        # if no TorchPhysics checkpoint callback exists, delete Modulus checkpoint and model files in the Modulus output directory            
        if not self.checkpoint_callback and not self.keep_output:  
            if os.path.isdir(os.path.abspath(os.getcwd()+'/'+self.outputdir_name)):
                shutil.rmtree(os.path.abspath(os.getcwd()+'/'+self.outputdir_name))                                          
        result = self.Msolver.parameters
            
        
        self.logger.removeHandler(self.ch)
        
        return result
        
    
        
