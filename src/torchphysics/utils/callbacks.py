import torch

from pytorch_lightning.callbacks import Callback

from .plotting.plot_functions import plot
from .user_fun import UserFunction


class WeightSaveCallback(Callback):
    """
    A callback to save the weights of a model during training. Can save
    the model weights before, during and after training. During training, only
    the model with minimal loss will be saved.

    Parameters
    ----------
    model : torch.nn.Module
        The model of which the weights should be saved.
    path : str
        The relative path of the saved weights.
    name : str
        A name that will become part of the file name of the saved weights.
    check_interval : int
        The callback will check for minimal loss every check_interval
        iterations. If negative, no weights will be saved during training.
    save_initial_model : False
        Whether the model should be saved before training as well.
    save_final_model: True
        Whether the model should always be saved after the last iteration.
    """
    def __init__(self, model, path, name, check_interval,
                 save_initial_model=False, save_final_model=True):
        super().__init__()
        self.model = model
        self.path = path
        self.name = name
        self.check_interval = check_interval
        self.save_initial_model = save_initial_model
        self.save_final_model = save_final_model

        self.current_loss = float('inf')

    def on_train_start(self, trainer, pl_module):
        if self.save_initial_model:
            torch.save(self.model.state_dict(), self.path+'/' + self.name + '_init.pt')
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (self.check_interval > 0 and batch_idx > 0) and ((batch_idx-1) % self.check_interval == 0):
            if trainer.logged_metrics['train/loss'] < self.current_loss:
                self.current_loss = trainer.logged_metrics['train/loss']
                torch.save(self.model.state_dict(),
                           self.path+'/' + self.name + '_min_loss.pt')

    def on_train_end(self, trainer, pl_module):
        if self.save_final_model:
            torch.save(self.model.state_dict(), self.path+'/' + self.name + '_final.pt')


class PlotterCallback(Callback):
    '''Object for plotting (logging plots) inside of tensorboard. 
    Can be passed to the pytorch lightning trainer.

    Parameters
    ----------
    plot_function : callable
        A function that specfices the part of the model that should be plotted.  
    point_sampler : torchphysics.samplers.PlotSampler
        A sampler that creates the points that should be used for the plot.
    log_interval : str, optional
        Name of the plots inside of tensorboard.
    check_interval : int, optional
        Plots will be saved every check_interval steps, if the plotter is used.
    angle : list, optional
        The view angle for surface plots. Standard angle is [30, 30]
    plot_type : str, optional
        Specifies how the output should be plotted. If no input is given, the method
        will try to use a fitting way, to show the data. See also plot-functions.
    kwargs:
        Additional arguments to specify different parameters/behaviour of
        the plot. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
        for possible arguments of each underlying object.
    '''
    def __init__(self, model, plot_function, point_sampler, log_name='plot', 
                 check_interval=200, angle=[30, 30], plot_type='', **kwargs):
        super().__init__()
        self.model = model
        self.check_interval=check_interval
        self.plot_function = UserFunction(plot_function)
        self.log_name = log_name
        self.point_sampler = point_sampler
        self.angle = angle
        self.plot_type = plot_type
        self.kwargs = kwargs

    def on_train_start(self, trainer, pl_module):
        self.point_sampler.sample_points(device=pl_module.device)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx, dataloader_idx):
        if batch_idx % self.check_interval == 0:
            fig = plot(model=self.model, plot_function=self.plot_function,
                       point_sampler=self.point_sampler, 
                       angle=self.angle, plot_type=self.plot_type,
                       device=pl_module.device, **self.kwargs)
            pl_module.logger.experiment.add_figure(tag=self.log_name,
                                                    figure=fig,
                                                    global_step=batch_idx)

    def on_train_end(self, trainer, pl_module):
        return


class TrainerStateCheckpoint(Callback):
    """
    A callback to saves the current state of the trainer (a PyTorch Lightning checkpoint),
    if the training has to be resumed at a later point in time.

    Parameters
    ----------
    path : str
        The relative path of the saved weights.
    name : str
        A name that will become part of the file name of the saved weights.
    check_interval : int, optional
        Checkpoints will be saved every check_interval steps. Default is 200.
    weights_only : bool, optional
        If only the model parameters should be saved. Default is false.

    Note
    ----
    To continue from the checkpoint, use `resume_from_checkpoint ="path_to_ckpt_file"` as an
    argument in the initialization of the trainer.

    The PyTorch Lightning checkpoint would save the current epoch and restart from it. 
    In TorchPhysics we dont use multiple epochs, instead we train with multiple iterations
    inside "one giant epoch". If the training is restarted, the trainer will always start
    from iteration 0 (essentially the last completed epoch). But all other states
    (model, optimizer, ...) will be correctly restored.
    """
    def __init__(self, path, name, check_interval=200, weights_only = False):
        super().__init__()
        self.path = path
        self.name = name
        self.check_interval = check_interval
        self.weights_only = weights_only


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.check_interval == 0:
            trainer.save_checkpoint(self.path + '/' + self.name + ".ckpt", 
                                    weights_only=self.weights_only)
