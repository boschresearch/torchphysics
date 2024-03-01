==============================
Applied TorchPhysics Tutorials
==============================
Here, we explain the library of TorchPhysics along the implementation of different 
examples. 

To start, we consider a heat equation problem of the form

.. math::
    \begin{align}
    \partial_t u(x,t) &= \Delta_x u(x,t) \text{ on } \Omega\times I, \\
    u(x, t) &= u_0 \text{ on } \Omega\times \{0\},\\
    u(x,t) &= h(t) \text{ at } \partial\Omega_{heater}\times I, \\
    \nabla_x u(x, t) \cdot n(x) &= 0 \text{ at } (\partial \Omega \setminus \partial\Omega_{heater}) \times I,
    \end{align}


that we will solve with PINNs. This example is a nice starting point for a new user and can 
be found here_. The notebook gives a lot of information about TorchPhysics and even repeats the
basic ideas of PINNs.

.. _here : https://github.com/boschresearch/torchphysics/blob/main/examples/tutorial/Introduction_Tutorial_PINNs.ipynb

A next step would be to make the problem more complicated, such that not a single solution
should be found, but a whole family of solutions for different functions :math:`h`.
As long as the different :math:`h` can be defined through some parameters, the solution operator 
can still be learned through PINNs. This is explained in this notebook_, which is really similar to
previous one and highlights the small aspects that have to be changed.

.. _notebook : https://github.com/boschresearch/torchphysics/blob/main/examples/tutorial/Tutorial_PINNs_Parameter_Dependency.ipynb

For more complex :math:`h` functions, we end up at the DeepONet. DeepONets can also be learned 
physics informed, which is demonstrated in this tutorial_.

.. _tutorial : https://github.com/boschresearch/torchphysics/blob/main/examples/tutorial/Introduction_Tutorial_DeepONet.ipynb

Similar examples, with a description of each step, can be found in the two notebooks `PINNs for Poisson`_ 
and `DRM for Poisson`_. The second notebook 
also uses the Deep Ritz Method instead of PINNs. 

More applications can be found on the `example page`_.

.. _`PINNs for Poisson`: https://github.com/boschresearch/torchphysics/blob/main/examples/tutorial/solve_pde.ipynb
.. _`DRM for Poisson`: https://github.com/boschresearch/torchphysics/blob/main/examples/tutorial/solve_pde_drm.ipynb
.. _`example page`: https://boschresearch.github.io/torchphysics/examples.html