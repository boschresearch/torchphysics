'''File contains different helper functions to get specific informations about
the computed solution.
'''
import time
import torch

from .user_fun import UserFunction
from ..problem.spaces import Points


def compute_min_and_max(model, sampler, evaluation_fn=lambda u:u, 
                        device='cpu', requieres_grad=False):
    '''Computes the minimum and maximum values of the model w.r.t. the given
    variables.

    Parameters
    ----------
    model : DiffEqModel
        A neural network of which values should be computed.
    sampler : torchphysics.samplers.PointSampler
        A sampler that creates the points where the model should be evaluated.
    evaluation_fn : callable
        A user-defined function that uses the neural network and creates the 
        desiered output quantity.
    device : str or torch device
        The device of the model.    
    track_gradients : bool
        Whether to track input gradients or not. 

    Returns
    -------
    float
        The minimum value computed.
    float 
        The maximum value computed.
    '''
    print('-- Start evaluation of minimum and maximum --')
    input_points = next(sampler)
    input_points._t.requires_grad = requieres_grad
    input_points._t.to(device)
    # eval. model
    inp_points_dict = input_points.coordinates
    start_model_eval = time.time()
    model_out = model(Points.from_coordinates(inp_points_dict))
    end_model_eval = time.time()
    # eval user function
    evaluation_fn = UserFunction(evaluation_fn)
    data_dict = {**model_out.coordinates, **inp_points_dict}
    start_func_eval = time.time()
    prediction = evaluation_fn(data_dict) 
    end_func_eval = time.time()
    max_pred = torch.max(prediction)
    min_pred = torch.min(prediction)
    print('Time to evaluate model:', end_model_eval - start_model_eval)
    print('Time to evaluate User-Function:', end_func_eval - start_func_eval)
    print('Found the values')
    print('Min:', min_pred)
    print('Max:', max_pred)
    return min_pred, max_pred
