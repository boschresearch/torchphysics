# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:29:24 2021

@author: inu2sh

1D Heat equation in Anlehnung an:
https://github.com/lululxvi/deepxde/issues/188

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import matplotlib.animation as animation
from matplotlib import cm


import time as time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(' --- 2D Heat difusion equation on unit square --- ')


#D = 18.8 # Eisen 
D = 1.18 #Granit

# Einheitsquadrat
x_min = 0
x_max = 10
y_min = 0
y_max = 10
dx = x_max - x_min
dy = y_max - y_min
# Zeit 
t_min = 0
t_max = 3
#
Tcool = 0.0
Thot = 100.

def pde(x, u):
    u_t = dde.grad.jacobian(u, x, j=2)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    return u_t - D * (u_xx + u_yy)


def ini_cond(x):
    s1 = np.sin(np.pi*x[:,0:1])
    return s1


def zeichnen(time_step):
    ax = plt.axes(projection ='3d')
    X = xx[:, :, time_step]
    Y = yy[:, :, time_step]
    Z = y_predicted[:, :, time_step]
    z_min = np.min(y_predicted)
    z_max = np.max(y_predicted)
    zeit = t[time_step]
    #ax.plot_wireframe(x, y, z )
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, shade=False, \
                    cmap="coolwarm", linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(t,x,y)')
    ax.set_title('Time = '+ np.str(zeit) + ' s')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max]) 
    ax.view_init(30, 45)
    plt.show(True)


# Geometry:
geom = dde.geometry.Rectangle(xmin=[x_min, y_min], xmax=[x_max, y_max])
timedomain = dde.geometry.TimeDomain(t_min, t_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(' --- boundary and initial conditions ---')
# abs(np.sin(np.pi*x[:,0:1]*np.sin(np.pi*x[:,1:2])))
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
#ic = dde.IC(geomtime, lambda x: 0.0 , lambda _, on_initial: on_initial)
ic = dde.IC(geomtime, lambda x: Thot*np.sin(np.pi*(x[:,0:1]-x_min)/(x_max-x_min))\
            *np.sin(np.pi*(x[:,1:2]-y_min)/(y_max-y_min)) , \
                lambda _, on_initial: on_initial)

# num_domain = 200000
# num_test = 1000000
Nx = 50 # [x_min, x_max] = 32 
Ny = 50 # [y_min, y_max] = 32
NB = 500
IB = 1000
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=Nx*Ny, num_boundary=NB, num_initial=IB)



net = dde.maps.FNN([3] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-2)
model.train(epochs=3000)
model.compile("L-BFGS-B", lr=1.e-3)
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# --- Prediction of solution on an equidistant grid --- 
resolution = 40
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
t = np.linspace(t_min, t_max, resolution, endpoint = True)
xx, yy, tt = np.meshgrid(x, y, t, indexing = 'ij')
Z = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T

print ('--- Prediction of solution ---')
t1 = time.time()
y_predicted = model.predict(Z).reshape(resolution, resolution, resolution)
f = model.predict(Z, operator=pde).reshape(resolution, resolution, resolution)
t2 = time.time()
print('--- prediction time for one step = ', (t2-t1)/resolution)

print('--- Lösung zeichnen ---')
for i in range(resolution):
    zeichnen(i)
    #print(t[i])
    time.sleep(0.1)
    plt.close()

print('Zeit = ', t[resolution-1])
print('Min. Temperatur = ', np.min(y_predicted[:,:,resolution-1]))    
print('Max. Temperatur = ', np.max(y_predicted[:,:,resolution-1]))
