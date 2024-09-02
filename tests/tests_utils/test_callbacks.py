import os
import torch
import pytorch_lightning as pl
import torchphysics as tp
from pytorch_lightning import loggers as pl_loggers
import shutil

def helper_setup():
    X = tp.spaces.R1("x")
    U = tp.spaces.R1("u")

    x_smapler = tp.samplers.RandomUniformSampler(
        tp.domains.Interval(X, 0, 1), 10
    )

    model = tp.models.FCN(X, U)

    def test_cond(u):
        return u 

    cond = tp.conditions.PINNCondition(model, x_smapler, test_cond)

    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.01)
    solver = tp.solver.Solver([cond], optimizer_setting=optim)

    return model, solver, x_smapler.domain


def helper_cleaner(path_to_file, delete_dict=False):
    try:
        if delete_dict:
            shutil.rmtree(path_to_file)
        else:
            os.remove(path_to_file)
    except OSError:
        raise AssertionError(f"File {path_to_file} does not exist! Callback did not work")


def test_weight_save_callback():
    model, solver, _ = helper_setup()
    save_callback = tp.WeightSaveCallback(model, "./tests", "test_weight",
                                          10,
                                          save_final_model=False)
    trainer = pl.Trainer( max_steps=2, callbacks=[save_callback])
    trainer.fit(solver)
    helper_cleaner("./tests/test_weight_min_loss.pt")


def test_weight_save_callback_end():
    model, solver, _ = helper_setup()
    save_callback = tp.WeightSaveCallback(model, "./tests", "test_weight",
                                          10)
    trainer = pl.Trainer( max_steps=2, callbacks=[save_callback])
    trainer.fit(solver)
    helper_cleaner("./tests/test_weight_min_loss.pt")
    helper_cleaner("./tests/test_weight_final.pt")


def test_weight_save_callback_start():
    model, solver, _ = helper_setup()
    save_callback = tp.WeightSaveCallback(model, "./tests", "test_weight",
                                          10, True, False)
    trainer = pl.Trainer( max_steps=2, callbacks=[save_callback])
    trainer.fit(solver)
    helper_cleaner("./tests/test_weight_min_loss.pt")
    helper_cleaner("./tests/test_weight_init.pt")


def test_plotter_callback():
    tensorboard_logger = pl_loggers.TensorBoardLogger('./tests/logdata')
    model, solver, domain = helper_setup()
    plot_sampler = tp.samplers.PlotSampler(domain, 100)
    plot_callback = tp.callbacks.PlotterCallback(model, lambda u : u, 
                                                 plot_sampler)
    trainer = pl.Trainer( max_steps=2, callbacks=[plot_callback], logger=tensorboard_logger)
    trainer.fit(solver)
    helper_cleaner("./tests/logdata", True)


def test_state_checkpoint():
    _, solver, _ = helper_setup()
    state_callback = tp.callbacks.TrainerStateCheckpoint("./tests", name="state")
    trainer = pl.Trainer( max_steps=2, callbacks=[state_callback])
    trainer.fit(solver)
    helper_cleaner("./tests/state.ckpt")