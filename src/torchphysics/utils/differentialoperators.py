'''File contains differentialoperators

NOTE: We aim to make the computation of differential operaotrs more efficient
      by building an intelligent framework that is able to keep already computed
      derivatives and therefore make the computations more efficient.
'''
import torch


def laplacian(model_out, *derivative_variable, grad=None):
    '''Computes the laplacian of a network with respect to the given variable

    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
        be computed
    grad : torch.tensor
        If the gradient has already been computed somewhere else, it is more
        efficient to use it again.

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the value of the sum of the second
        derivatives (laplace) w.r.t the row of the input variable.
    '''
    laplacian = torch.zeros((*model_out.shape[:-1], 1),
                             device=model_out.device)
    for vari in derivative_variable:
        if grad is None or len(derivative_variable) > 1:
            grad = torch.autograd.grad(model_out.sum(), vari, create_graph=True)[0]
        # We have to check if the model is linear w.r.t. the variable, or else we get an err
        # when we compute the second derivative. If it is linear we can just return zeros
        if grad.grad_fn is None:
            continue
        for i in range(vari.shape[-1]):
            D2u = torch.autograd.grad(grad.narrow(-1, i, 1).sum(),
                                      vari, create_graph=True)[0]
            laplacian += D2u.narrow(-1, i, 1)
    return laplacian


def grad(model_out, *derivative_variable):
    '''Computes the gradient of a network with respect to the given variable.
    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
        be computed
    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the the first
        derivatives (gradient) w.r.t the row of the input variable.
    '''
    grad = []
    for vari in derivative_variable:
        new_grad = torch.autograd.grad(model_out.sum(), vari,
                                       create_graph=True)[0]
        grad.append(new_grad)
    return torch.column_stack(grad)

"""
def grad(model_out, *derivative_variable):
    '''Computes the gradient of a network with respect to the given variable.

    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
        be computed

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the the first
        derivatives (gradient) w.r.t the row of the input variable.
    '''
    grad = []
    # store identity matrix (necessary in batched grad computation) only once
    eye = None
    dim = list(range(len(model_out.shape)))
    assert model_out.shape[-1] == 1
    for vari in derivative_variable:
        if vari.shape[:-1] == model_out.shape[:-1]:
            new_grad = torch.autograd.grad(model_out.sum(dim),
                                           vari,
                                           create_graph=True)[0]
        else:
            assert vari.shape[:-1] == model_out.shape[1:-1]
            if eye is None:
                eye = torch.eye(model_out.shape[0], device=vari.device)
            new_grad = torch.autograd.grad(model_out.sum(dim[1:]),
                                           vari,
                                           grad_outputs=(eye,),
                                           is_grads_batched=True,
                                           create_graph=True)[0]
        grad.append(new_grad)
    return torch.cat(grad, dim=-1)
"""

def normal_derivative(model_out, normals, *derivative_variable):
    '''Computes the normal derivativ of a network with respect to the given variable
    and normal vectors.

    Parameters
    ----------
    model_out : torch.tensor
        The (scalar) output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
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
    gradient = grad(model_out, *derivative_variable)
    normal_derivatives = gradient*normals
    return normal_derivatives.sum(dim=-1, keepdim=True)
    

def div(model_out, *derivative_variable):
    '''Computes the divergence of a network with respect to the given variable.
    Only for vector valued inputs, for matices use the function matrix_div.
    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
        be computed. Have to be in a consistent ordering, if for example the output 
        is u = (u_x, u_y) than the variables has to passed in the order (x, y)
    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the divergence
        of the model w.r.t the row of the input variable.
    '''
    divergence = torch.zeros((derivative_variable[0].shape[0], 1),
                             device=derivative_variable[0].device)
    var_dim = 0
    for vari in derivative_variable:
        for i in range(vari.shape[1]):
            Du = torch.autograd.grad(model_out.narrow(1, var_dim + i, 1).sum(),
                                     vari, create_graph=True)[0]
            divergence += Du.narrow(1, i, 1)
        var_dim += i + 1
    return divergence

"""
def div(model_out, *derivative_variable):
    '''Computes the divergence of a network with respect to the given variable.
    Only for vector valued inputs, for matices use the function matrix_div.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    derivative_variable : torch.tensor
        The input tensor of the variables in which respect the derivatives have to
        be computed. Have to be in a consistent ordering, if for example the output 
        is u = (u_x, u_y) than the variables has to passed in the order (x, y)

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the divergence
        of the model w.r.t the row of the input variable.
    '''
    divergence = torch.zeros((*derivative_variable[0].shape[:-1], 1),
                              device=derivative_variable[0].device)
    var_dim = 0
    dim = list(range(len(model_out.shape)))
    model_out_sum = None
    model_out_partial_sum = None
    for vari in derivative_variable:
        if vari.shape[:-1] == model_out.shape[:-1]:
            # standard case (e.g. PINN)
            if model_out_sum is None:
                model_out_sum = model_out.sum(dim[:-1])
            if vari.shape[-1] > 1:
                eye_out = torch.eye(vari.shape[-1], device=vari.device)
                new_div = torch.autograd.grad(model_out_sum[var_dim:var_dim+vari.shape[-1]],
                                              vari,
                                              grad_outputs=(eye_out,),
                                              is_grads_batched=True,
                                              create_graph=True)[0]
            else:
                new_div = torch.autograd.grad(model_out_sum.narrow(-1, var_dim, 1),
                                              vari, create_graph=True)[0]
        else:
            # advanced, e.g. efficient DeepONet case
            assert vari.shape[:-1] == model_out.shape[1:-1]
            eye = torch.eye(model_out.shape[0]*vari.shape[-1])
            if model_out_partial_sum is None:
                model_out_partial_sum = model_out.sum(dim[1:-1])
            if vari.shape[-1] > 1:
                new_div = torch.autograd.grad(model_out_partial_sum[:, var_dim:var_dim+vari.shape[-1]].reshape(-1, 1),
                                              vari,
                                              grad_outputs=(eye,),
                                              is_grads_batched=True,
                                              create_graph=True)[0].reshape(model_out.shape[0], vari.shape[-1])
            else:
                new_div = torch.autograd.grad(model_out_partial_sum[:, var_dim:var_dim+1],
                                              vari,
                                              grad_outputs=(eye,),
                                              is_grads_batched=True,
                                              create_graph=True)[0]
        var_dim += vari.shape[-1]
        divergence = divergence + new_div
    return divergence
"""

def jac(model_out, *derivative_variable):
    '''Computes the jacobian of a network output with
    respect to the given input.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor in which respect the jacobian should be computed.
    derivative_variable : torch.tensor
        The input tensor in which respect the jacobian should be computed.

    Returns
    ----------
    torch.tensor
        A Tensor of shape (b, m, n), where every row contains a jacobian.
    '''
    Du_rows = []
    for i in range(model_out.shape[1]):
        Du_i = []
        for vari in derivative_variable:
            Du_i.append(torch.autograd.grad(model_out[:, i].sum(),
                                            vari, create_graph=True)[0])
        Du_rows.append(torch.cat(Du_i, dim=1))
    Du = torch.stack(Du_rows, dim=1)
    return Du


def rot(model_out, *derivative_variable):
    '''Computes the rotation/curl of a 3-dimensional vector field (given by a
    network output) with respect to the given input.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of shape (b, 3) in which respect the roation should be
        computed.
    derivative_variable : torch.tensor
        The input tensor of shape (b, 3) in which respect the rotation should be
        computed.

    Returns
    ----------
    torch.tensor
        A Tensor of shape (b, 3), where every row contains a rotation/curl vector for a
        given batch element.
    '''
    """
    assert model_out.shape[1] == 3 and derivative_variable.shape[1] == 3, ""
        Rotation: the given in- and output should both be batches of
        3 dimensional data.
        ""
    """
    jacobian = jac(model_out, *derivative_variable)
    rotation = torch.zeros((len(derivative_variable[0]), 3))
    rotation[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2] 
    rotation[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0] 
    rotation[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1] 
    return rotation


def partial(model_out, *derivative_variables):
    '''Computes the (n-th, possibly mixed) partial derivative of a network output with
    respect to the given variables.

    Parameters
    ----------
    model_out : torch.tensor
        The output tensor of the neural network
    derivative_variables : torch.tensor(s)
        The input tensors in which respect the derivatives should be computed. If n
        tensors are given, the n-th (mixed) derivative will be computed.

    Returns
    ----------
    torch.tensor
        A Tensor, where every row contains the values of the computed partial 
        derivative of the model w.r.t the row of the input variable.
    '''
    du = model_out
    for inp in derivative_variables:
        du = torch.autograd.grad(du.sum(),
                                 inp,
                                 create_graph=True)[0]
        if du.grad_fn is None:
            return torch.zeros_like(inp)
    return du


def convective(deriv_out, convective_field, *derivative_variable):
    '''Computes the convective term :math:`(v \\cdot \\nabla)u` that appears e.g. in
    material derivatives. Note: This is not the whole material derivative.

    Parameters
    ----------
    deriv_out : torch.tensor
        The vector or scalar field :math:`u` that is convected and should be
        differentiated.
    convective_field: torch.tensor
        The flow vector field :math:`v`. Should have the same dimension as
        derivative_variable.
    derivative_variable : torch.tensor
        The spatial variable in which respect deriv_out should be differentiated.

    Returns
    ----------
    torch.tensor
        A vector or scalar (+batch-dimension) Tensor, that contains the convective
        derivative.
    '''
    jac_x = jac(deriv_out, *derivative_variable)
    return torch.bmm(jac_x, convective_field.unsqueeze(dim=2)).squeeze(dim=2)


def sym_grad(model_out, *derivative_variable):
    """Computes the symmetric gradient: :math:`0.5(\nabla u + \nabla u^T)`.

    Parameters
    ----------
    model_out : torch.tensor
        The vector field :math:`u` that should be differentiated.
    derivative_variable : torch.tensor
        The spatial variable in which respect model_out should be differentiated.

    Returns
    ----------
    torch.tensor
        A Tensor of matrices of the form (batch, dim, dim), containing the 
        symmetric gradient.
    """
    jac_matrix = jac(model_out, *derivative_variable)
    return 0.5 * (jac_matrix + torch.transpose(jac_matrix, 1, 2))


def matrix_div(model_out, *derivative_variable):
    """Computes the divergence for matrix/tensor-valued functions.

    Parameters
    ----------
    model_out : torch.tensor
        The (batch) of matirces that should be differentiated.
    derivative_variable : torch.tensor
        The spatial variable in which respect should be differentiated.

    Returns
    ----------
    torch.tensor
        A Tensor of vectors of the form (batch, dim), containing the 
        divegrence of the input.
    """
    div_out = torch.zeros((len(model_out), model_out.shape[1]), 
                          device=model_out.device)
    for i in range(model_out.shape[1]):
        # compute divergence of matrix by computing the divergence 
        # for each row
        current_row = model_out.narrow(1, i, 1).squeeze(1)
        div_out[:, i:i+1] = div(current_row, *derivative_variable)
    return div_out
