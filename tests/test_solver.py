import pytest
import torch
import pytorch_lightning as pl

from torchphysics.solver import PINNModule
from torchphysics.setting import Setting
from torchphysics.models.fcn import SimpleFCN
from torchphysics.utils.plot import Plotter
from torchphysics.problem.condition import DirichletCondition
from torchphysics.problem.variables import Variable
from torchphysics.problem.parameters import Parameter


# Helper functions
def _create_model():
    model = SimpleFCN(variable_dims={'x': 2},
                      solution_dims={'u': 1},
                      depth=1,
                      width=5)
    return model


def _create_dummy_trainer(log=False):
    trainer = pl.Trainer(gpus=None,
                         num_sanity_val_steps=0,
                         benchmark=False,
                         check_val_every_n_epoch=20,
                         max_epochs=0,
                         logger=log,
                         checkpoint_callback=False)
    return trainer


def _create_whole_dummy_setting():
    model = _create_model()
    setup = Setting()
    trainer = _create_dummy_trainer()
    solver = PINNModule(model=model, 
                        optimizer=torch.optim.Adam, 
                        lr = 3)  
    trainer.datamodule = setup
    solver.trainer = trainer
    return solver, setup, trainer


def _add_dummy_variable(setup, name, train=True):
    x = Variable(name='x', domain=None)
    cond = DirichletCondition(dirichlet_fun=None,
                              name=name, 
                              norm=torch.nn.MSELoss())
    if train:
        x.add_train_condition(cond)
    else:
        x.add_val_condition(cond)
    setup.add_variable(x)


# Start test of PINNModule
def test_create_pinn_module():
    solver = PINNModule(model=None, 
                        optimizer=torch.optim.Adam, 
                        lr = 3)
    assert solver.model is None
    assert solver.optimizer == torch.optim.Adam
    assert solver.lr == 3
    assert solver.log_plotter is None
    assert solver.optim_params == {}
    assert solver.scheduler is None


def test_forward_pinn_module():
    solver = PINNModule(model=_create_model(), 
                        optimizer=torch.optim.Adam, 
                        lr = 3)
    input_dic = {'x': torch.ones((4, 2))}
    out = solver(input_dic)
    assert isinstance(out, dict)
    assert torch.is_tensor(out['u'])
    assert out['u'].shape == (4, 1)


def test_input_dim_pinn_module():
    solver = PINNModule(model=_create_model(), 
                        optimizer=torch.optim.Adam, 
                        lr = 3)
    assert solver.input_dim == 2


def test_output_dim_pinn_module():
    solver = PINNModule(model=_create_model(), 
                        optimizer=torch.optim.Adam, 
                        lr = 3)
    assert solver.output_dim == 1


def test_to_device_pinn_module_without_trainer():
    solver = PINNModule(model=_create_model(), 
                        optimizer=torch.optim.Adam, 
                        lr = 3)  
    out = solver.to('cpu')
    assert out.device.type == 'cpu' 


def test_to_device_pinn_module_with_trainer():
    solver, _, _ = _create_whole_dummy_setting()
    out = solver.to('cpu')
    assert out.device.type == 'cpu'  


def test_serialize_pinn_module_without_trainer():
    model = _create_model()
    solver = PINNModule(model=model, 
                        optimizer=torch.optim.Adam, 
                        lr = 3)  
    out = solver.serialize()
    assert out['name'] == 'PINNModule'
    assert out['model'] == model.serialize()
    assert out['problem'] is None
    assert out['optimizer']['name'] == 'Adam'
    assert out['optimizer']['lr'] == 3
    assert out['optim_params'] == {}


def test_serialize_pinn_module_with_trainer():
    solver, setup, _ = _create_whole_dummy_setting()
    out = solver.serialize()
    assert out['name'] == 'PINNModule'
    assert out['model'] == solver.model.serialize()
    assert out['problem'] == setup.serialize()
    assert out['optimizer']['name'] == 'Adam'
    assert out['optimizer']['lr'] == 3
    assert out['optim_params'] == {}


def test_on_train_start_without_logger():
    solver = PINNModule(model=None, 
                        optimizer=torch.optim.Adam, 
                        lr = 3) 
    solver.on_train_start()


def test_configure_optimizer_of_pinn_module():
    solver, _, _ = _create_whole_dummy_setting()
    opti = solver.configure_optimizers()
    assert isinstance(opti, torch.optim.Optimizer)
    for p in opti.param_groups:
        assert p['lr'] == 3 


def test_configure_optimizer_of_pinn_module_with_scheduler():
    solver, _, _ = _create_whole_dummy_setting()
    solver.scheduler = {'class': torch.optim.lr_scheduler.ExponentialLR, 
                        'args': {'gamma': 3}}
    opti, scheduler = solver.configure_optimizers()
    assert isinstance(opti[0], torch.optim.Optimizer)
    for p in opti[0].param_groups:
        assert p['lr'] == 3
    assert isinstance(scheduler[0], torch.optim.lr_scheduler._LRScheduler)#

"""Test dont work in GitHub....
def test_training_step_of_pinn_module():
    solver, setup, _ = _create_whole_dummy_setting()
    _add_dummy_variable(setup, 'test')
    data = {'x': torch.tensor([[2.0, 1.0], [3.0, 0.0]], requires_grad=True), 
            'target': torch.tensor([[2.0], [3.0]])}
    batch = {'x_test': data}
    out = solver.training_step(batch, 0)
    assert isinstance(out, torch.Tensor)


def test_training_step_of_pinn_module_with_parameters():
    solver, setup, _ = _create_whole_dummy_setting()
    _add_dummy_variable(setup, 'test')
    setup.add_parameter(Parameter([1, 0], name='D'))
    setup.add_parameter(Parameter(0, name='k'))
    data = {'x': torch.tensor([[2.0, 1.0], [3.0, 0.0]], requires_grad=True), 
            'target': torch.tensor([[2.0], [3.0]])}
    batch = {'x_test': data}
    out = solver.training_step(batch, 0)
    assert isinstance(out, torch.Tensor)


def test_training_step_of_pinn_module_with_missing_data():
    solver, setup, _ = _create_whole_dummy_setting()
    _add_dummy_variable(setup, 'test')
    _add_dummy_variable(setup, 'test_2')
    data = {'x': torch.tensor([[2.0, 1.0], [3.0, 0.0]], requires_grad=True), 
            'target': torch.tensor([[2.0], [3.0]])}
    batch = {'x_test': data}
    with pytest.raises(KeyError):
        _ = solver.training_step(batch, 0)


def test_validation_step_of_pinn_module():
    solver, setup, _ = _create_whole_dummy_setting()
    _add_dummy_variable(setup, 'test', False)
    data = {'x': torch.tensor([[2.0, 1.0], [3.0, 0.0]], requires_grad=True), 
            'target': torch.tensor([[2.0], [3.0]])}
    batch = {'x_test': data}
    solver.validation_step(batch, 0)
"""