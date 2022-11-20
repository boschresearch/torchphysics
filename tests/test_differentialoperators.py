import pytest
import torch
import numpy as np
from torchphysics.utils.differentialoperators import (laplacian, 
                                                      grad, 
                                                      normal_derivative, 
                                                      div, 
                                                      jac, 
                                                      rot,
                                                      partial, 
                                                      convective, 
                                                      sym_grad, 
                                                      matrix_div)

# Test laplace-operator
def function(a):
    out = 0
    for i in range(len(a)):
        out += a[i]**2
    return out


def test_laplacian_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = a**2
    l = laplacian(output, a)
    assert l.shape[0] == 1
    assert l.shape[1] == 1
    assert l.detach().numpy()[0] == 4


def test_laplacian_for_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 3.4], [1.3, 2], [0, 0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4, 4, 4, 4])


def test_laplacian_in_1D():
    a = torch.tensor([[1.0], [2.0], [1.3]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [2, 2, 2])


def test_laplacian_in_3D():
    a = torch.tensor([[1.0, 3.4, 1.0], [2.0, 0, 0], [1.3, 9, 1]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [6, 6, 6])


def test_laplacian_with_grad_input():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return torch.sin(2*a[0]) + a[1]**2 + b[0]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    g = grad(output.unsqueeze(-1), a)
    l = laplacian(output.reshape(-1, 1), a, grad=g)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[-4*np.sin(2)+2], [-4*np.sin(4)+2]])


def test_laplacian_with_grad_input_2():
    def f(input):
        out = torch.zeros((input.shape[0], 2))
        out[:, :1] = input[:, :1]**2 + input[:, 2:]**2 + input[:, 1:2]
        out[:, 1:] = torch.sin(input[:, 2:]*input[:, 1:2])
        return out
    x = torch.tensor([[1.0, 2], [1, 1], [3, 4], [3, 0]], requires_grad=True)
    t = torch.tensor([[3.0], [0], [1], [2]], requires_grad=True)
    inp = torch.cat((x, t), dim=1)
    output = f(inp)
    jacobi = jac(output, inp)
    assert jacobi.shape == (4, 2, 3)
    l_1 = laplacian(output, x, grad=jacobi[:, 0, :2])
    l_2 = laplacian(output[:, :1], x)
    assert torch.equal(l_1, l_2)


def test_laplacian_for_complexer_function_1():
    a = torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 0], [0, 0, 0], [1.0, 0, 4.0]],
                     requires_grad=True)
    def function1(a):
        return a[0]**2 + a[1]**3 + 4*a[2]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[32], [8], [2], [98]])


def test_laplacian_for_complexer_function_2():
    a = torch.tensor([[1.0, 1.0], [2.0, 0], [0, 0],
                      [0, 4.0], [2, 2]], requires_grad=True)
    def function1(a):
        return a[0]**3 + torch.sin(a[1])
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[6-np.sin(1)], [12], [0],
                                            [-np.sin(4)], [12-np.sin(2)]])


def test_laplacian_for_two_inputs_one_linear():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return 2*a[0]**2 + a[1]**2 + b[0]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [6, 6])  
    l = laplacian(output.reshape(-1, 1), b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [0, 0])  



def test_laplacian_for_two_not_linear_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1]**2 + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4, 4])  
    l = laplacian(output.reshape(-1, 1), b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[6], [3]]) 



def test_laplacian_multiply_varibales():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [2]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 * a[1]**2 * b[0]**2
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[4], [32]])  
    l = laplacian(output.reshape(-1, 1), b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[2], [0]])    



def test_laplacian_with_chain_rule():
    a = torch.tensor([[1.0, 1], [2.0, 1]], requires_grad=True)
    def function1(a):
        return torch.sin(2.0 * (torch.sin(a[0]))) * a[1]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[-0.97203], [-0.22555]], atol=1e-04)  


def test_laplacian_with_tanh():
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 0, 1.0]], requires_grad=True)
    def function1(a):
        return torch.tanh(a[0]**2 * a[1]**3 + a[2]**2)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output.reshape(-1, 1), a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[-0.0087], [-1.7189]], atol=1e-04)  


def test_laplacian_for_two_variables_at_the_same_time():
    x = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    y = torch.tensor([[1.0], [2.0]], requires_grad=True)
    def function1(x, y):
        return x[0]**2 * x[1]**3 + y[0]**2
    output = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        output[i] = function1(x[i], y[i])
    l = laplacian(output.reshape(-1, 1), x, y)
    assert l.shape[0] == x.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[30], [28]])          

# Test gradient
def test_gradient_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = function(a[0]).unsqueeze(-1).unsqueeze(0)
    g = grad(output, a)
    assert g.shape[0] == 1
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [2, 2]).all()


def test_gradient_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2, 0], [3, 1]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output.unsqueeze(-1), a)
    assert g.shape[0] == 3
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [[2, 2], [4, 0], [6, 2]]).all()   


def test_gradient_1D():
    a = torch.tensor([[1.0], [2.0], [0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output.unsqueeze(-1), a)
    assert g.shape[0] == 3
    assert g.shape[1] == 1
    assert np.equal(g.detach().numpy(), [[2], [4], [0]]).all()


def test_gradient_3D():
    a = torch.tensor([[1.0, 5, 2], [2.0, 2.0, 2.0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output.unsqueeze(-1), a)
    assert g.shape[0] == 2
    assert g.shape[1] == 3
    assert np.equal(g.detach().numpy(), [[2, 10, 4], [4, 4, 4]]).all()


def test_gradient_mixed_input():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1] + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    g = grad(output.unsqueeze(-1), a)
    assert g.shape[0] == a.shape[0]
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [[2, 1], [4, 1]]).all()  
    g = grad(output.unsqueeze(-1), b)
    assert g.shape[0] == b.shape[0]
    assert g.shape[1] == 1
    assert np.equal(g.detach().numpy(), [[3], [3/4]]).all() 


def test_gradient_for_two_variables_at_the_same_time():
    x = torch.tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    y = torch.tensor([[1.0], [2.0]], requires_grad=True)
    def function1(x, y):
        return x[0]**2 * x[1]**3 + y[0]**2
    output = torch.zeros(x.shape[0])
    for i in range(x.shape[0]):
        output[i] = function1(x[i], y[i])
    g = grad(output.unsqueeze(-1), x, y)
    assert g.shape[0] == x.shape[0]
    assert g.shape[1] == 3
    assert np.allclose(g.detach().numpy(), [[16, 12, 2], [4, 12, 4]])     

# Test normal derivative
def test_normal_derivative_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = function(a[0])
    normal = torch.tensor([[1.0, 0]])
    n = normal_derivative(output.unsqueeze(-1).unsqueeze(0), normal, a)
    assert n.shape[0] == 1
    assert n.shape[1] == 1
    assert np.equal(n.detach().numpy(), [2]).all()


def test_normal_derivative_for_many_inputs():
    a = torch.tensor([[1.0, 1.0], [0, 1], [2, 3]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    normals = torch.tensor([[1.0, 0], [1.0, 0], [np.cos(np.pi/4), np.sin(np.pi/4)]])
    n = normal_derivative(output.unsqueeze(-1), normals, a)
    assert n.shape[0] == 3
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [0],
                                            [4*np.cos(np.pi/4)+6*np.sin(np.pi/4)]])


def test_normal_derivative_3D():
    a = torch.tensor([[1.0, 1.0, 1.0], [0, 1, 2]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    normals = torch.tensor([[1.0, 0, 0], [1.0, 0, 1.0]])
    n = normal_derivative(output.unsqueeze(-1), normals, a)
    assert n.shape[0] == 2
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [4]])


def test_normal_derivative_complexer_function():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 3.0]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + torch.sin(a[1]) + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    normals = torch.tensor([[1.0, 0], [1.0/np.sqrt(2), 1.0/np.sqrt(2)]])
    n = normal_derivative(output.unsqueeze(-1), normals, a)
    assert n.shape[0] == a.shape[0]
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [1/np.sqrt(2)*(4+np.cos(0))]])
    n = normal_derivative(output.unsqueeze(-1), normals, b)
    assert n.shape[0] == b.shape[0]
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[3], [27/np.sqrt(2)]])


# Test divergence
def div_function(x):
    return x**2


def test_div_one_input():
    a = torch.tensor([[1.0, 0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape == (1, 1)
    d = d.detach().numpy()
    assert d[0] == 2


def test_div_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape == (2, 1)
    d = d.detach().numpy()
    assert d[0] == 4
    assert d[1] == 6


def test_div_in_3D():
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape[0] == 2
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert d[0] == 8
    assert d[1] == 6


def test_div_for_complexer_function_1():
    def f(x):
        out = x**2
        out[:, :1] *= x[:, 1:]
        return out
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [5.0, 2.0]], requires_grad=True)
    output = f(a)
    d = div(output, a)
    assert d.shape[0] == 3
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert d[0] == 4
    assert d[1] == 6
    assert d[2] == 24


def test_div_for_complexer_function_2():
    def f(x):
        out = x**2
        out[:, :1] = torch.sin(x[:, 1:] * x[:, :1])
        return out
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [5.0, 2.0]], requires_grad=True)
    output = f(a)
    d = div(output, a)
    assert d.shape[0] == 3
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert np.isclose(d[0], 1*(2+np.cos(1)))
    assert np.isclose(d[1], 1*(2+np.cos(2)))
    assert np.isclose(d[2], 2*(2+np.cos(10)))


def test_div_for_two_variables_at_the_same_time():
    def f(x, y):
        return torch.cat((x, y), dim=1)
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    b = torch.tensor([[1.0], [2.0]], requires_grad=True)
    output = f(a, b)
    d = div(output, a, b)
    assert d.shape[0] == a.shape[0]
    assert d.shape[1] == 1
    assert np.allclose(d.detach().numpy(), [[3], [3]])  


# Test Jacobi-Matrix
def jac_function(x):
    out = x**2
    out[:, :1] += x[:, 1:2] 
    return out


def test_jac_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (1, 2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1], [0, 2]]).all()


def test_jac_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [0.0, 3]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (3, 2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1], [0, 2]]).all()
    assert np.isclose(d[1], [[4, 1], [0, 2]]).all()
    assert np.isclose(d[2], [[0, 1], [0, 6]]).all()


def test_jac_in_3D():
    a = torch.tensor([[1.0, 1.0, 0.0], [2.0, 1.0, 2.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1, 0], [0, 2, 0], [0, 0, 0]]).all()
    assert np.isclose(d[1], [[4, 1, 0], [0, 2, 0], [0, 0, 4]]).all()


def test_jac_for_complexer_function():
    def jac_function(x):
        out = x**3
        out[:,:1] += torch.sin(x[:,1:2])
        out[:,1:2] *= x[:,2:]
        return out
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 3.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[3,  np.cos(1), 0], [0, 6, 1], [0, 0, 12]]).all()
    assert np.isclose(d[1], [[12, np.cos(1), 0], [0, 9, 1], [0, 0, 27]]).all()


def test_jac_for_complexer_function_2():
    def jac_function(x):
        out = x**2
        out[:,:1] += torch.sin(x[:,1:2]*x[:,2:])
        out[:,1:2] *= x[:,2:]
        out[:,2:] *= torch.exp(x[:,:1])
        return out
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 3.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2,  np.cos(2)*2, np.cos(2)], [0, 4, 1],
                             [4*np.exp(1), 0, 4*np.exp(1)]]).all()
    assert np.isclose(d[1], [[4,  np.cos(3)*3, np.cos(3)], [0, 6, 1],
                             [9*np.exp(2), 0, 6*np.exp(2)]]).all()


def test_jac_for_two_variables_at_the_same_time():
    def f(x, y):
        out = torch.zeros((len(x), 2))
        out[:, :1] = x**2 * y
        out[:, 1:] = y + torch.exp(x) 
        return out
    a = torch.tensor([[0.0], [3.0]], requires_grad=True)
    b = torch.tensor([[1.0], [2.0]], requires_grad=True)
    output = f(a, b)
    d = jac(output, a, b)
    assert d.shape == (2, 2, 2)
    assert torch.allclose(d[0], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert torch.allclose(d[1], torch.tensor([[12.0, 9.0], [torch.exp(a[1]), 1.0]]))


# Test rot
def rot_function(x):
    out = torch.zeros((len(x), 3))
    out[:, :1] += x[:, 1:2] 
    out[:, 1:2] -= x[:, :1]
    return out


def test_rot_one_input():
    a = torch.tensor([[1.0, 1.0, 2.0]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (1, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [0, 0, -2]).all()


def test_rot_many_inputs():
    a = torch.tensor([[1, 1, 2.0], [0, 1.0, 0], [1.0, 3.0, 4]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    for i in range(3):
        assert np.isclose(d[i], [0, 0, -2]).all()


def test_rot_for_complexer_function():
    def rot_function(x):
        out = torch.zeros((len(x), 3))
        out[:, 1:2] -= x[:, :1]**2
        return out       
    a = torch.tensor([[-1, 1, 2.0], [1.0, 1.0, 0], [2.0, 3.0, 4]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [0, 0, 2]).all()
    assert np.isclose(d[1], [0, 0, -2]).all()
    assert np.isclose(d[2], [0, 0, -4]).all()


def test_rot_for_complexer_function_2():
    def rot_function(x):
        out = torch.zeros((len(x), 3))
        out[:, :1]  = torch.sin(x[:,1:2]*x[:,2:])
        out[:, 1:2] = -x[:,:1]**2
        out[:, 2:]  = x[:,:1] + x[:,1:2]
        return out       
    a = torch.tensor([[-1, 1, 2.0], [1.0, 1.0, 0]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (2, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [1, np.cos(2)-1, 2-2*np.cos(2)]).all()
    assert np.isclose(d[1], [1, np.cos(0)-1, -2]).all()


def test_rot_for_two_variables_at_the_same_time():
    def rot_function(x, y):
        out = torch.zeros((len(x), 3))
        out[:, :1]  = torch.sin(x[:,1:]*y[:,:1])
        out[:, 1:2] = -x[:,:1]**2
        out[:, 2:]  = x[:,:1] + x[:,1:]
        return out   
    a = torch.tensor([[-1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    b = torch.tensor([[2.0], [0.0]], requires_grad=True)
    output = rot_function(a, b)
    d = rot(output, a, b)
    assert d.shape == (2, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [1, np.cos(2)-1, 2-2*np.cos(2)]).all()
    assert np.isclose(d[1], [1, np.cos(0)-1, -2]).all()


# Test partial
def part_function(x, y, t):
    out = x**2 * t + y * torch.sin(t)
    return out


def test_partial_one_input():
    x = torch.tensor([[1.0]], requires_grad=True)
    y = torch.tensor([[1.0]], requires_grad=True)
    t = torch.tensor([[2.0]], requires_grad=True)
    output = part_function(x, y, t)
    p = partial(output, x)
    assert p.shape == (1, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], 4)
    p = partial(output, y)
    assert p.shape == (1, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], np.sin(2))


def test_partial_many_inputs():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[1.0], [3.0]], requires_grad=True)
    t = torch.tensor([[2.0], [0.0]], requires_grad=True)
    output = part_function(x, y, t)
    p = partial(output, x)
    assert p.shape == (2, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], 4)
    assert np.isclose(d[1], 0)
    p = partial(output, y)
    assert p.shape == (2, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], np.sin(2))
    assert np.isclose(d[1], 0)


def test_partial_mixed_derivative():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[1.0], [3.0]], requires_grad=True)
    t = torch.tensor([[2.0], [0.0]], requires_grad=True)
    output = part_function(x, y, t)
    p = partial(output, x, t, x)
    assert p.shape == (2, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], 2)
    assert np.isclose(d[1], 2)


def test_partial_mixed_derivative_2():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[1.0], [3.0]], requires_grad=True)
    t = torch.tensor([[2.0], [0.0]], requires_grad=True)
    output = part_function(x, y, t)
    p = partial(output, y, t, t)
    assert p.shape == (2, 1)
    d = p.detach().numpy()
    assert np.isclose(d[0], -np.sin(2))
    assert np.isclose(d[1], 0)


def test_partial_repeated_gives_0():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[1.0], [3.0]], requires_grad=True)
    t = torch.tensor([[2.0], [0.0]], requires_grad=True)
    output = part_function(x, y, t)
    p = partial(output, x, x, x)
    assert p.shape == (2, 1)
    d = p.detach().numpy()
    assert np.allclose(d[0], [[0], [0]])


# Test convective
def convec_function(x):
    out = torch.zeros((len(x), 3))
    out[:, :1] += x[:, :1]**2 
    out[:, 1:2] += x[:, :1]
    out[:, 2:] += x[:, 1:2] * x[:, 2:]
    return out


def test_convective_one_input():
    a = torch.tensor([[1.0, 1.0, 2.0]], requires_grad=True)
    output = convec_function(a)
    d = convective(output, output, a)
    assert d.shape == (1, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [2, 1, 4]).all()


def test_convective_many_inputs():
    a = torch.tensor([[1, 1, 2.0], [0, 1.0, 0], [1.0, 3.0, 4]], requires_grad=True)
    output = convec_function(a)
    d = convective(output, output, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [2, 1, 4]).all()
    assert np.isclose(d[1], [0, 0, 0]).all()
    assert np.isclose(d[2], [2, 1, 40]).all()


def test_convective_for_different_conv_field():
    a = torch.tensor([[1, 1, 2.0], [0, 1.0, 0], [1.0, 3.0, 4]], requires_grad=True)
    output = convec_function(a)
    d = convective(output, a, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [2, 1, 4]).all()
    assert np.isclose(d[1], [0, 0, 0]).all()
    assert np.isclose(d[2], [2, 1, 24]).all()


def test_convective_in_2D():
    a = torch.tensor([[1, 1], [0, 1.0]], requires_grad=True)
    def func(x):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x[:, :1]**2 
        out[:, 1:2] += x[:, 1:]
        return out
    output = func(a)
    d = convective(output, a, a)
    assert d.shape == (2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [2, 1]).all()
    assert np.isclose(d[1], [0, 1]).all()


def test_convective_in_for_two_variables_at_the_same_time():
    a = torch.tensor([[1.0], [0]], requires_grad=True)
    b = torch.tensor([[1.0], [1.0]], requires_grad=True)
    def func(x, y):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x[:, :1]**2 
        out[:, 1:2] += y[:, :1]
        return out
    output = func(a, b)
    d = convective(output, torch.cat((a, b), dim=1), a, b)
    assert d.shape == (2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [2, 1]).all()
    assert np.isclose(d[1], [0, 1]).all()


def test_sym_grad():
    a = torch.tensor([[1.0, 2.0], [0, 2.0]], requires_grad=True)
    def func(x):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x[:, :1]*x[:, 1:]
        out[:, 1:] += 2*x[:, 1:]
        return out
    output = func(a)
    s_grad = sym_grad(output, a)
    assert s_grad.shape == (2, 2, 2)
    assert torch.allclose(s_grad[0], torch.tensor([[2.0, 0.5], [0.5, 2.0]]))
    assert torch.allclose(s_grad[1], torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_sym_grad_complexer_fn():
    a = torch.tensor([[1.0, 2.0], [0, 2.0], [1.0, 3.0]], requires_grad=True)
    def func(x):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x[:, :1]**2 * x[:, 1:] + x[:, :1]
        out[:, 1:] += x[:, 1:] * torch.sin(x[:, :1])
        return out
    output = func(a)
    s_grad = sym_grad(output, a)
    assert s_grad.shape == (3, 2, 2)
    off_diag = 0.5*(1 + 2*np.cos(1.0))
    assert torch.allclose(s_grad[0], torch.tensor([[5.0, off_diag],
                                                   [off_diag, np.sin(1.0)]], 
                                                   dtype=torch.float32))
    off_diag = 0.5*(2*np.cos(0.0))
    assert torch.allclose(s_grad[1], torch.tensor([[1.0, off_diag],
                                                   [off_diag, np.sin(0.0)]], 
                                                   dtype=torch.float32))
    off_diag = 0.5*(1.0 + 3*np.cos(1.0))
    assert torch.allclose(s_grad[2], torch.tensor([[7.0, off_diag],
                                                   [off_diag, np.sin(1.0)]], 
                                                   dtype=torch.float32))


def test_divergence_for_matrix():
    a = torch.tensor([[1.0, 2.0], [0, 2.0]], requires_grad=True)
    def func(x):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x[:, :1]*x[:, 1:]**2
        out[:, 1:] += 2*x[:, 1:] + x[:, :1]**3
        return out
    output = func(a)
    o_jac = jac(output, a)
    m_div = matrix_div(o_jac, a)
    assert m_div.shape == (2, 2)
    assert torch.allclose(m_div[0], torch.tensor([2.0, 6.0]))
    assert torch.allclose(m_div[1], torch.tensor([0.0, 0.0]))


def test_divergence_for_matrix_for_complexer_fn():
    a = torch.tensor([[1.0, 2.0], [0, 2.0], [1.0, 5.0]], requires_grad=True)
    def func(x):
        out = torch.zeros((len(x), 2))
        out[:, :1] += torch.exp(x[:, :1])*x[:, 1:]**2
        out[:, 1:] += 2*x[:, 1:] + torch.sin(x[:, :1]**3)
        return out
    output = func(a)
    o_jac = sym_grad(output, a)
    m_div = matrix_div(o_jac, a)
    assert m_div.shape == (3, 2)
    expected_out = torch.zeros_like(a)
    expected_out[:, :1] = torch.exp(a[:, :1]) * (a[:, 1:]**2 + 1.0)
    expected_out[:, 1:] = torch.exp(a[:, :1]) * a[:, 1:]  \
        + 3.0*a[:, :1]*torch.cos(a[:, :1]**3) - 4.5*a[:, :1]**4*torch.sin(a[:, :1]**3)
    assert torch.allclose(m_div, expected_out)


def test_divergence_for_matrix_with_multiple_variables():
    a = torch.tensor([[1.0], [0]], requires_grad=True)
    b = torch.tensor([[2.0], [2.0]], requires_grad=True)
    def func(x, y):
        out = torch.zeros((len(x), 2))
        out[:, :1] += x*y**2
        out[:, 1:] += 2*y + x**3
        return out
    output = func(a, b)
    o_jac = jac(output, a, b)
    m_div = matrix_div(o_jac, a, b)
    assert m_div.shape == (2, 2)
    assert torch.allclose(m_div[0], torch.tensor([2.0, 6.0]))
    assert torch.allclose(m_div[1], torch.tensor([0.0, 0.0]))