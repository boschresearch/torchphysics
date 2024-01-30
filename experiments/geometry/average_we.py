import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt

average_exmaples = 100
# stays the same:
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output is 1D
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])
scaling = 0.01
steps = 4
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps]
xx = xx.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
x_points = x_points.reshape(-1, 2).astype(np.float32)
x_points = tp.spaces.Points(torch.tensor(x_points), X) 
pde_sampler = tp.samplers.DataSampler(x_points)
a = torch.zeros(1)
kernel = [[0, 1, 0], [1, 0, -1], [0, -1, 0]]
# boundary condition:
def bound_residual(u):
    return u**2
def bound_residual_pinn(u):
    return u
bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=int(2052/steps)).make_static()
# pde 
def energy_residual(u, x):
    return 1/2 * a * torch.sum(tp.utils.grad(u, x)**2, dim=1, keepdim=True) - 1.0

def pde_residual(u, x):
    u_grad = tp.utils.grad(u, x)
    conv = torch.sum(u_grad * a_grad, dim=1).reshape(-1, 1)
    return (conv + a*tp.utils.laplacian(u, x, grad=u_grad) + 1.0) * weights

# load data
input_data = np.load("/home/tomfre/Desktop/torchphysics/experiments/SIAM-Darcy/f_data.npy")[:average_exmaples, ::steps, ::steps]
output_data = np.load("/home/tomfre/Desktop/torchphysics/experiments/SIAM-Darcy/u_data.npy")[:average_exmaples]

input_data = scaling * input_data
output_data = torch.tensor(output_data)

l2_error_array = np.zeros(average_exmaples)

for i in range(average_exmaples):
    print("current step:", i)
    # reset model
    #model = tp.models.DeepRitzNet(input_space=X, output_space=U, depth=3, width=60)
    model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50, 20))
    #model = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 36, 15))
    # set current data
    a = input_data[i]
    kernel = [[0, 0, 0], [1, 0, -1], [0, 0, 0]]
    a_x = scipy.signal.convolve2d(a, kernel, mode="same", boundary='symm') * 513/steps
    kernel = [[0, 1, 0], [0, 0, 0], [0, -1, 0]]
    a_y = scipy.signal.convolve2d(a, kernel, mode="same", boundary='symm') * 513/steps

    a = torch.tensor(a)
    a_x, a_y = torch.tensor(a_x), torch.tensor(a_y)
    weights = torch.ones_like(a)
    weights[a_x != 0] = steps/513
    weights[a_y != 0] = steps/513
    a = a.reshape(-1, 1)
    a_grad = torch.column_stack((a_x.reshape(-1, 1), a_y.reshape(-1, 1)))
    weights = weights.reshape(-1, 1)
    a = a.to('cuda')
    a_grad = a_grad.to('cuda')
    weights = weights.to('cuda')
    # define conditions
    ## DRM
    #bound_cond = tp.conditions.DeepRitzCondition(module=model, sampler=bound_sampler, 
    #                                             integrand_fn=bound_residual, weight=100)
    #pde_cond = tp.conditions.DeepRitzCondition(model, pde_sampler, energy_residual)
    ## PINN/QRES
    bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler, 
                                             residual_fn=bound_residual_pinn, weight=100)
    pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler, residual_fn=pde_residual)
    # start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)       
    trainer.fit(solver)
    # LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.25, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=12000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0) 
    trainer.fit(solver)

    # compute current error:
    model.to('cpu')
    u_out = model(xx)
    u_out = (u_out.as_tensor * scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(u_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i] = l2_rel
    print("In step ", i, "had error:", l2_rel)
    np.save("experiments/SIAM-Darcy/weight_129.npy", l2_error_array)

print("Average error:", np.sum(l2_error_array)/len(l2_error_array))