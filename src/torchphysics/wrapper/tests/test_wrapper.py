import pytest
import torchphysics as tp
import torch
import pytorch_lightning as pl
import os
import shutil
from tensorboard.backend.event_processing import event_accumulator

from torchphysics.wrapper import TPModulusWrapper, ModulusSolverWrapper
from torchphysics.utils import PointsDataLoader
from torchphysics.problem.spaces import Points, R1
from omegaconf import DictConfig
from modulus.sym.solver import Solver


def _create_dummies():
    fcn = tp.FCN(tp.spaces.R1('x'), tp.spaces.R1('u'))
    ps = tp.samplers.RandomUniformSampler(tp.domains.Interval(tp.spaces.R1('x'), 0, 1), 
                                          n_points=10)
    cond = tp.conditions.PINNCondition(fcn, ps, lambda u: u)    
    opti = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.1,)
    solver = tp.Solver(train_conditions=[cond],optimizer_setting=opti)    
    trainer = pl.Trainer(max_steps=15)     
    
    return fcn, trainer, solver

def _create_dummies_with_params():  
    model = tp.FCN(tp.spaces.R1('x'), tp.spaces.R1('u'))
    ps = tp.samplers.RandomUniformSampler(tp.domains.Interval(tp.spaces.R1('x'), 0, 1), 
                                          n_points=10)
    p = tp.models.parameter.Parameter(init=1.0, space=tp.spaces.R1('D'))
    cond1 = tp.conditions.PINNCondition(model, ps, lambda u,D: u*D,parameter=p)     

    loader = PointsDataLoader((Points(torch.tensor([[0.0], [2.0]]), R1('x')),
                               Points(torch.tensor([[0.0], [4.0]]), R1('u'))),batch_size=2)
                              

    cond2 = tp.conditions.DataCondition(module=model, dataloader=loader, norm=2)
    
    opti = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.1,)
    solver = tp.Solver(train_conditions=[cond1,cond2],optimizer_setting=opti)    
    trainer = pl.Trainer(max_steps=15)     
    return trainer, solver, p

def test_TPModulusWrapper_creation():
    _, trainer,solver = _create_dummies()        
    aggregator_args = {"alpha": 0.5}
    scheduler_args = {"T_max": 999}
    wrapper = TPModulusWrapper(trainer=trainer, solver=solver, lambda_weighting=["sdf"],aggregator="GradNorm", aggregator_args=aggregator_args,scheduler="CosineAnnealingLR", scheduler_args=scheduler_args,outputdir_name="my_output_dir",keep_output=True,confdir_name="my_conf_dir")
    assert isinstance(wrapper, TPModulusWrapper)

    assert wrapper.cfg.loss._target_ == "modulus.sym.loss.aggregator.GradNorm"    
    assert wrapper.cfg.loss["alpha"] == 0.5
    assert wrapper.cfg.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert isinstance(wrapper.cfg, DictConfig)
    assert wrapper.cfg.scheduler.T_max == 999
    assert wrapper.cfg.loss.alpha == 0.5
    assert wrapper.cfg.optimizer.lr == 0.1
    assert wrapper.cfg.network_dir == "my_output_dir"
    assert wrapper.cfg.training.max_steps == 15
    assert wrapper.keep_output == True
    assert wrapper.cfg.optimizer._target_ == "torch.optim.Adam"

    assert wrapper.Msolver.lambda_weighting_vals == ["sdf"]
    assert isinstance(wrapper.Msolver,ModulusSolverWrapper)    
    assert isinstance(wrapper.slv,Solver)
    
    assert  os.path.isdir(os.path.abspath(os.getcwd()+'/my_conf_dir'))


    solver.optimizer_setting.scheduler_class=torch.optim.lr_scheduler.ExponentialLR
    solver.optimizer_setting.scheduler_args={'gamma':0.8}
    wrapper = TPModulusWrapper(trainer=trainer, solver=solver, lambda_weighting=["sdf"],aggregator="GradNorm", aggregator_args=aggregator_args,outputdir_name="my_output_dir",keep_output=False,confdir_name="my_conf_dir")
    assert wrapper.cfg.scheduler._target_ == "torch.optim.lr_scheduler.ExponentialLR"
    assert wrapper.cfg.scheduler.gamma == 0.8



def test_callbacks():
    model, trainer,solver = _create_dummies()
    
    trainer.callbacks.append(tp.utils.WeightSaveCallback(model= model,path=os.getcwd(),name='test',check_interval = 5, save_initial_model=True,
                                    save_final_model = True))
    
    with pytest.warns(UserWarning, match="The option check_interval of the WeightSaveCallback with check for minimial loss is not supported by Modulus. Only initial and final model state saves."):
        wrapper = TPModulusWrapper(trainer=trainer, solver=solver,outputdir_name="my_output_dir",confdir_name="my_conf_dir")   
        wrapper.train()
    
    assert os.path.isfile(os.path.abspath(os.getcwd()+'/test_init.pt'))
    assert os.path.isfile(os.path.abspath(os.getcwd()+'/test_final.pt'))
   
    model, trainer,solver = _create_dummies()
    plot_sampler = tp.samplers.PlotSampler(tp.domains.Interval(tp.spaces.R1('x'), 0, 1), 
                                          n_points=50)
    trainer.callbacks.append(tp.utils.PlotterCallback(model=model,plot_function=lambda u: u,point_sampler=plot_sampler,
                       log_name='plot_u', plot_type='plot',check_interval=5))
    wrapper = TPModulusWrapper(trainer=trainer, solver=solver,keep_output=True, outputdir_name="my_output_dir",confdir_name="my_conf_dir") 
    wrapper.train()
    # path to tensorboard-events-file
    log_dir = os.path.join(os.getcwd(), 'my_output_dir')
    event_file = None

    # Check if tensorboard-events-file exists
    for file in os.listdir(log_dir):
        if file.startswith("events.out.tfevents"):
            event_file = os.path.join(log_dir, file)
            break

    assert event_file is not None, "tensorboard-events-file not found."

    # Usage of EventAccumulator to load events file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Check if plot is contained
    tags = ea.Tags()['images']
    assert 'Inferencers/plot_u/' in tags, "Plot 'plot_u' not found in tensorboard-events-file."

    


def test_train_function():
    _, trainer,solver = _create_dummies()
    wrapper = TPModulusWrapper(trainer=trainer, solver=solver,outputdir_name="my_output_dir",confdir_name="my_conf_dir")    
    assert wrapper.train()==[]    
    assert os.path.isdir(os.path.abspath(os.getcwd()+'/my_output_dir'))
    
    trainer,solver, p = _create_dummies_with_params()
    wrapper = TPModulusWrapper(trainer=trainer, solver=solver,keep_output=True, outputdir_name="my_output_dir",confdir_name="my_conf_dir") 
    wrapper.train()
    assert abs(p.as_tensor- 1.0) >1e-15
    

    

def teardown_module(module):
    """This method is called after test completion."""
    conf_dir = os.path.abspath(os.getcwd() + '/my_conf_dir')
    if os.path.isdir(conf_dir):
        shutil.rmtree(conf_dir)
    output_dir = os.path.abspath(os.getcwd() + '/my_output_dir')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    if os.path.isfile(os.path.abspath(os.getcwd()+'/test_init.pt')):
        os.remove(os.path.abspath(os.getcwd()+'/test_init.pt'))
    if os.path.isfile(os.path.abspath(os.getcwd()+'/test_final.pt')):
        os.remove(os.path.abspath(os.getcwd()+'/test_final.pt'))