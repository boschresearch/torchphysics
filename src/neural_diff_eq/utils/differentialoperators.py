'''File contains differentialoperators
'''
import torch


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
    laplacian = torch.zeros((deriv_variable_input.shape[0], 1),
                            device=deriv_variable_input.device)
    Du = torch.autograd.grad(model_out.sum(), deriv_variable_input,
                             create_graph=True)[0]
    # We have to check if the model is linear w.r.t. the variable, or else we get an err
    # when we compute the second derivative. If it is linear we can just return zeros
    if Du.grad_fn is None:
        return laplacian
    for i in range(deriv_variable_input.shape[1]):
        D2u = torch.autograd.grad(Du.narrow(1, i, 1).sum(),
                                  deriv_variable_input, create_graph=True)[0]
        laplacian += D2u.narrow(1, i, 1)
    return laplacian


def grad(model_out, deriv_variable_input):
    '''Computes the gradient of a network with respect to the given variable.

    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
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


def normal_derivative(model_out, deriv_variable_input, normals):
    '''Computes the normal derivativ of a network with respect to the given variable
    and normal vectors.

    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
    deriv_variable_input : torch.tensor
        The input tensor of the variable in which respect the derivatives have to
        be computed
    normals : torch.tensor
        The normal vectors at the points where the derivative has to be computed.
        In the form: normals = tensor([normal_1, normal_2, ...]

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the normal
        derivatives w.r.t the row of the input variable.
    '''
    gradient = grad(model_out, deriv_variable_input)
    normal_derivatives = gradient*normals
    return normal_derivatives.sum(dim=1, keepdim=True)


def div(model_out, deriv_variable_input):
    '''Computes the divergence of a network with respect to the given variable.

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
        A Tensor, where every row contains the values of the divergence
        of the model w.r.t the row of the input variable.
    '''
    divergence = torch.zeros((deriv_variable_input.shape[0], 1),
                             device=deriv_variable_input.device)
    for i in range(deriv_variable_input.shape[1]):
        Du = torch.autograd.grad(model_out.narrow(1, i, 1).sum(),
                                 deriv_variable_input, create_graph=True)[0]
        divergence += Du.narrow(1, i, 1)
    return divergence


def jac(model_out, deriv_variable_input):
    '''Computes the jacobian of a network output with
    respect to the given input.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor in which respect the jacobian should be computed.
    deriv_variable_input : torch.tensor
        The input tensor in which respect the jacobian should be computed.

    Returns
    ----------
    torch.tensor
        A Tensor of shape (b, m, n), where every row contains a jacobian.
    '''
    Du_rows = []
    for i in range(model_out.shape[1]):
        Du_rows.append(torch.autograd.grad(model_out[:, i].sum(),
                                           deriv_variable_input,
                                           create_graph=True)[0])
    Du = torch.stack(Du_rows, dim=1)
    return Du


def rot(model_out, deriv_variable_input):
    '''Computes the rotation/curl of a 3-dimensional vector field (given by a
    network output) with respect to the given input.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of shape (b, 3) in which respect the roation should be
        computed.
    deriv_variable_input : torch.tensor
        The input tensor of shape (b, 3) in which respect the rotation should be
        computed.

    Returns
    ----------
    torch.tensor
        A Tensor of shape (b, 3), where every row contains a rotation/curl vector for a
        given batch element.
    '''
    """
    assert model_out.shape[1] == 3 and deriv_variable_input.shape[1] == 3, ""
        Rotation: the given in- and output should both be batches of
        3 dimensional data.
        ""
    """
    jacobian = jac(model_out, deriv_variable_input)
    rotation = torch.zeros((len(deriv_variable_input), 3))
    rotation[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2] 
    rotation[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0] 
    rotation[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1] 
    return rotation


def partial(model_out, *deriv_variable_inputs):
    '''Computes the (n-th, possibly mixed) partial derivative of a network output with
    respect to the given variables.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    deriv_variable_inputs : torch.tensor(s)
        The input tensors in which respect the derivatives should be computed. If n
        tensors are given, the n-th (mixed) derivative will be computed.

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the computed partial 
        derivative of the model w.r.t the row of the input variable.
    '''
    du = model_out
    for inp in deriv_variable_inputs:
        du = torch.autograd.grad(du.sum(),
                                 inp,
                                 create_graph=True)[0]
        if du.grad_fn is None:
            return torch.zeros_like(inp)
    return du
