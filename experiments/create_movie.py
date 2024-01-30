import matplotlib.pyplot as plt 
import matplotlib.tri as mtri
from matplotlib import cm
import matplotlib as mpl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torchphysics as tp
import numpy as np
import torch
import pytorch_lightning as pl

radius = 5.0
widht, height = 0.5, 4.0
t_0, t_end = 0.0, 1.0
a_0, a_end = np.pi, 2*np.pi # rotation speed
k_cool, k_hot = 270, 330 # temperature values
mu = 2.0 # viscosity
lamb = 0.5 # heat diffusion

def start_fn(t, a):
    return 1 - torch.exp(-a*t)
# dont want to jump from initial temperature to k_hot
def heat_up_fn(t, a):
    return k_cool + (k_hot - k_cool) * start_fn(t, a)


X = tp.spaces.R2('x') # space
T = tp.spaces.R1('t') # time
A = tp.spaces.R1('a') # rotation speed
U = tp.spaces.R2('u') # velocity
P = tp.spaces.R1('p') # pressure
K = tp.spaces.R1('k') # temperature

def rotation_function(a, t):
    # rotate clockwise and use t*(1-e^-at) -> velocity has a smooth start up
    return - a * t * start_fn(t, a)

circle = tp.domains.Circle(X, [0.0, 0.0], radius)
rod = tp.domains.Parallelogram(X, [-widht, -height], [widht, -height], [-widht, height])
rod = tp.domains.Rotate.from_angles(rod, rotation_function)
omega = circle - rod
t_int = tp.domains.Interval(T, t_0, t_end)
a_int = tp.domains.Interval(A, a_0, a_end)

# model for velocity and pressure
ac_fn = tp.models.AdaptiveActivationFunction(torch.nn.Tanh())
model_u = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=U, hidden=(100,100,100,100,100,100,100), 
                  activations=ac_fn)
)
model_p = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=P, hidden=(80,80,80,80,80,80))
)
ac_fn_temp = tp.models.AdaptiveActivationFunction(torch.nn.Tanh())
model_k = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=K, hidden=(100,100,100,100,100,100,100), 
                  activations=ac_fn_temp)
)


## load models:
model_u.load_state_dict(torch.load('/home/tomfre/Desktop/torchphysics/final_u.pt'))
model_k.load_state_dict(torch.load('/home/tomfre/Desktop/torchphysics/final_temp.pt'))

# constrain for the velocity:
# outer boundary u = 0 and initialy u = 0
def constrain_fn_u(u, x, t, a):
    time_scale = start_fn(t, a)
    rot_speed = radius*a*torch.column_stack((torch.cos(-a*t*time_scale), 
                                            torch.sin(-a*t*time_scale)))
    rot_speed *= (1 - torch.exp(-a*t) + a*t*torch.exp(-a*t))
    distance = x[:, :1]**2 + x[:, 1:]**2
    u_con = rot_speed * u * (1 - distance / radius**2)
    return u_con
# constrain for the pressure:
# initialy p = 0
def constrain_fn_p(p, t, a):
    p_con = p * start_fn(t, a)
    return p_con
# constrain for the temperature:
# initialy k = k_cool and on outer boundary k = k_cool
def constrain_fn_k(k, x, t, a):
    distance = x[:, :1]**2 + x[:, 1:]**2
    k_con = (k_hot - k_cool) * start_fn(t, a) * k * (1.0 - distance/radius**2) + k_cool
    return k_con



mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.figsize'] = [17.5, 15.0]
mpl.rcParams['font.size'] = 32
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 32
mpl.rcParams['figure.dpi'] = 200

fig = plt.figure()
a_value = 2*np.pi
time_steps = torch.linspace(0, 1, 50)

points = np.load("/home/tomfre/Desktop/torchphysics/experiments/vertex_points.npy")
connection = np.load("/home/tomfre/Desktop/torchphysics/experiments/connections.npy")
a = torch.tensor(a_value * np.ones_like(points[:, :1]), dtype=torch.float32)
points = torch.tensor(points, dtype=torch.float32)

for i in range(len(time_steps)):
    # rotate points
    angle = rotation_function(torch.tensor(time_steps[i]), torch.tensor(a_value))
    matrix = torch.tensor([[torch.cos(angle - np.pi/2.0), -torch.sin(angle - np.pi/2.0)],
                           [torch.sin(angle - np.pi/2.0), torch.cos(angle - np.pi/2.0)]])
    matrix = matrix.reshape(-1, 2, 2)
    p = torch.matmul(matrix, points.reshape(-1, 2, 1)).squeeze(-1)
    # model evaluation
    t = time_steps[i] * torch.ones_like(points[:, :1])
    tp_points = tp.spaces.Points(torch.column_stack((p, t, a)), X*T*A)
    out = model_k(tp_points).as_tensor.detach()
    out = constrain_fn_k(out, p, t, a)
    # plot
    fig = plt.figure()
    ax = plt.gca()
    plt.grid()
    plt.xlim((-5.4, 5.4))
    plt.ylim((-5.4, 5.4))
    # helper for correct colorbar
    con = [ax.scatter([0, 0], [0, 1], c=[k_cool, k_hot],
                    vmin=k_cool, vmax=k_hot, cmap=cm.jet)]
    plt.colorbar(con[0])
    con[0].remove()
    triangulation = mtri.Triangulation(x=p[:, 0], y=p[:, 1], triangles=connection)
    ax.tricontourf(triangulation, out.flatten(), cmap=cm.jet, 
                        vmin=k_cool, vmax=k_hot)
    props = dict(boxstyle='round', facecolor='w', alpha=0.9)
    time_str = "%0.2f" % time_steps[i].item()
    ax.text(0.65, 0.95, r"$\omega = 2\pi$, " + 't = '+ time_str, transform=ax.transAxes, fontsize=32,
            verticalalignment='top', bbox=props, ha='left')
    #plt.legend([r"$\omega = 2\pi$, " + 't = '+ str(time_steps[i])])
    plt.tight_layout()
    omega_str = "%0.2f" % a_value
    plt.savefig("PicFolder/temperature_omega_" + omega_str + "_idx_" + str(i)+ ".png")
    plt.close(fig)