import torchphysics as tp 
import torch

### Define Spaces
X = tp.spaces.R2('x')
U = tp.spaces.R1('u')

### Define Domain
omega = tp.domains.Parallelogram(X, [0,0], [1,0], [0,1])

### Define Sampler (the location for the training)
inner_sampler = tp.samplers.RandomUniformSampler(omega, 
                                        n_points=15000)  

bound_sampler = tp.samplers.GridSampler(omega.boundary, 
                                        n_points=5000)

### Define the neural network
model = tp.models.FCN(input_space=X, 
                      output_space=U,
                      hidden=(20,20,20))

### Implement the math equations / condition
def pde_residual(u, x):
    return tp.utils.laplacian(u, x) - 1.0
                              
def boundary_residual(u, x):
    return u - 0.0

### Combine the model, sampler and condition 
boundary_cond = tp.conditions.PINNCondition(model,
                                    bound_sampler, 
                                    boundary_residual)

pde_cond = tp.conditions.PINNCondition(model, 
                                    inner_sampler, 
                                    pde_residual)

### Start training
optim = tp.OptimizerSetting(torch.optim.Adam, lr=0.001)
solver = tp.solver.Solver([boundary_cond, pde_cond], optimizer_setting=optim)

import pytorch_lightning as pl
trainer = pl.Trainer(gpus=1, # use one GPU 
                     max_steps=3000) # iteration number
trainer.fit(solver)

### Plot solution
plot_sampler = tp.samplers.PlotSampler(plot_domain=omega, 
                                       n_points=2000)
fig = tp.utils.plot(model, lambda u : u, plot_sampler)