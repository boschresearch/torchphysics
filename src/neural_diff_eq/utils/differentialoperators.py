'''File contains differentialoperators
'''
import torch
from torch._C import device


def laplacian(model_out, deriv_variable_input):
    '''Computes the laplacian of a network with respect to the given variable

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    deriv_variable_input : torch.tensor
        The input tensor of the variable in which respect the derivatives have to
        be computed

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the value of the sum of the second
        derivatives (laplace) w.r.t the row of the input variable.
    '''
    laplacian = torch.zeros((deriv_variable_input.shape[0], 1), device=deriv_variable_input.device)
    Du = torch.autograd.grad(
        model_out.sum(), deriv_variable_input, create_graph=True)[0]
    # We have to check if the model is linear w.r.t. the variable, or else we get an err
    # when we compute the second derivative. If it is linear we can just return zeros
    if Du.grad_fn is None:
        return laplacian
    for i in range(deriv_variable_input.shape[1]):
        D2u = torch.autograd.grad(Du.narrow(1, i, 1).sum(),
                                  deriv_variable_input, create_graph=True)[0]
        laplacian += D2u.narrow(1, i, 1)
    return laplacian


def gradient(model_out, deriv_variable_input):
    '''Computes the gradient of a network with respect to the given variable.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    deriv_variable_input : torch.tensor
        The input tensor of the variable in which respect the derivatives have to
        be computed

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the the first
        derivatives (gradient) w.r.t the row of the input variable.
    '''
    grad = torch.autograd.grad(
        model_out.sum(), deriv_variable_input, create_graph=True)[0]
    return grad
