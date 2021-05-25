'''File contains differentialoperators
'''
import torch

def laplacian(model_out, deriv_variable_input):
    '''Computes the laplacian of a network in respect to the given variable 
    model_out : torch.tensor
        The output tensor of the neural network 
    deriv_variable_input : torch.tensor
        The input tensor of the variable in which respect the derivatives have to
        be computed
    '''
    laplacian = torch.zeros((deriv_variable_input.shape[0],1))
    Du = torch.autograd.grad(model_out.sum(), deriv_variable_input, create_graph=True)[0]
    # If function is linear in the variable we have to check, or else we get an error
    if Du.grad_fn is None:
        return laplacian
    D2u = torch.autograd.grad(Du.sum(), deriv_variable_input, create_graph=True)[0]
    for i in range(deriv_variable_input.shape[1]):
        laplacian += D2u.narrow(1,i,1)
    return laplacian