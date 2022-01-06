import torch

import torchphysics as tp


def _create_dummy_problem():
    fcn = tp.FCN(tp.spaces.R1('x'), tp.spaces.R1('u'))
    ps = tp.samplers.RandomUniformSampler(tp.domains.Interval(tp.spaces.R1('x'), 0, 1), 
                                          n_points=10)
    cond = tp.conditions.PINNCondition(fcn, ps, lambda u: u)
    return cond


def test_create_optimizer_setting():
    opti = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.1, 
                               scheduler_class=torch.optim.lr_scheduler.ExponentialLR)
    assert opti.lr == 0.1
    assert opti.optimizer_class == torch.optim.Adam
    assert opti.scheduler_args == {}
    assert opti.optimizer_args == {}
    assert opti.scheduler_class == torch.optim.lr_scheduler.ExponentialLR


def test_create_solver():
    solver = tp.Solver(train_conditions=[])
    assert isinstance(solver.train_conditions, torch.nn.ModuleList)
    assert isinstance(solver.val_conditions, torch.nn.ModuleList)
    assert isinstance(solver.optimizer_setting, tp.OptimizerSetting)


def test_config_optimizers():
    cond = _create_dummy_problem()
    opi = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.1)
    solver = tp.Solver(train_conditions=[cond], optimizer_setting=opi)
    solver_opi = solver.configure_optimizers()
    assert isinstance(solver_opi, torch.optim.Adam)
    for p in solver_opi.param_groups:
        assert p['lr'] == 0.1 


def test_config_optimizers_with_lr_scheduler():
    cond = _create_dummy_problem()
    opi = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.1, 
                              scheduler_class=torch.optim.lr_scheduler.ExponentialLR, 
                              scheduler_args={'gamma': 3},
                              scheduler_frequency=2)
    solver = tp.Solver(train_conditions=[cond], optimizer_setting=opi)
    solver_opi, scheduler = solver.configure_optimizers()
    assert isinstance(solver_opi[0], torch.optim.Adam)
    for p in solver_opi[0].param_groups:
        assert p['lr'] == 0.1 
    assert isinstance(scheduler[0], dict)
    assert isinstance(scheduler[0]['scheduler'], torch.optim.lr_scheduler.ExponentialLR)
    assert scheduler[0]['frequency'] == 2