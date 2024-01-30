import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch
import h5py

average_exmaples = 100
# stays the same:
X = tp.spaces.R2('x')
U = tp.spaces.R1('u')
F = tp.spaces.R1('f')
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])
scaling = 100
noise = 0.1
steps = 8
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps].reshape(-1, 2).astype(np.float32)
xx = xx.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
x_points = tp.spaces.Points(torch.tensor(x_points), X)
pde_sampler = tp.samplers.DataSampler(x_points)

def pde_residual(f, u, x):
    return tp.utils.laplacian(u, x) + f

# load data
#path = '/localdata/nick/poisson/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#hf = h5py.File(path, "r")
#output_data = hf['test_in'][:average_exmaples, :, :]
#input_data = hf['test_out'][:average_exmaples, :, :]
output_data = np.load("/home/tomfre/Desktop/torchphysics/experiments/SIAM-Poisson/f_data.npy")[:average_exmaples]
input_data = np.load("/home/tomfre/Desktop/torchphysics/experiments/SIAM-Poisson/u_data.npy")[:average_exmaples, ::steps, ::steps]
# shapes:
# 64*64, 128*128, 256*256
input_data = torch.tensor(input_data).reshape(average_exmaples, 65*65)
input_data *= scaling
output_data = torch.tensor(output_data)
#hf.close()

l2_error_array = np.zeros((average_exmaples, 4)) # l2, accru, u_noisy, u_correct

for i in range(average_exmaples):
    print("current step:", i)
    # reset model
    #model_u = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50))
    model_u = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 36))
    #model_f = tp.models.FCN(input_space=X, output_space=F, hidden=(50, 50, 50))
    model_f = tp.models.QRES(input_space=X, output_space=F, hidden=(36, 36, 36))
    parallel_model = tp.models.Parallel(model_u, model_f)
    # set current data
    data = input_data[i].unsqueeze(-1)
    data += (noise * torch.max(data)) * torch.randn_like(data)
    data = tp.spaces.Points(data, U)
    # define conditions
    pde_cond = tp.conditions.PINNCondition(module=parallel_model, sampler=pde_sampler, 
                                           residual_fn=pde_residual)
    data_loader = tp.utils.PointsDataLoader((x_points.as_tensor, data), batch_size=50000)
    data_condition = tp.conditions.DataCondition(module=model_u,
                                                dataloader=data_loader,
                                                norm=2, 
                                                use_full_dataset=False, 
                                                weight=10000) 
    # start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)         
    trainer.fit(solver)
    # LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.2, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    data_condition.use_full_dataset = True
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=10000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)         
    trainer.fit(solver)

    # compute current error:
    model_f.to('cpu')
    model_u.to('cpu')
    f_out = model_f(xx)
    f_out = (f_out.as_tensor / scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(f_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i, 0] = l2_rel
    l2_error_array[i, 1] = torch.max(diff) / torch.max(torch.abs(output_data[i]))
    print("In step ", i, "had errors:", l2_error_array[i, :])
    np.save("experiments/SIAM-Poisson/nosie_l2_error_10P.npy", l2_error_array)

print("Average error:", np.sum(l2_error_array[:, 0])/len(l2_error_array))