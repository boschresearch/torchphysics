========
Examples
========
Here we want to mention and link some examples and applications 
that, in our opinion, nicely present the functionalities of TorchPhysics
and the PINN idea.

More examples can be found under the `examples-folder`_.

.. _`examples-folder`: https://github.com/boschresearch/torchphysics/tree/main/examples

Poisson problem
===============
One of the simplest applications is the forward solution of a Poisson equation: 

.. math::

   \begin{align}
   \Delta u &= 4.25\pi^2 u, \text{ in } \Omega = [0, 1] \times [0, 1] \\
   u &= \sin(\frac{\pi}{2} x_1)\cos(2\pi x_2), \text{ on } \partial \Omega
   \end{align}

This problem is part of the tutorial and is therefore explained with alot of details. 
The corresponding implementation can be found here_.

.. _here : tutorial/solve_pde.html

Learning parameter dependencies
===============================
A natural extension of the PINN approach is to learn parameter dependencies, 
that appear in the differential equation.
A simple example would be the problem:

.. math::

   \begin{align*}
      \partial_x u &= k u,  \text{ in } [0, 1] \\
      u(0) &= 1
   \end{align*}

where we want to train a family of solutions for :math:`k \in [0, 2]`. We
want to find the function :math:`u(x, k) = e^{kx}`.
Implemented is this example in: `simple-parameter-dependency-notebook`_ 

.. _`simple-parameter-dependency-notebook`: https://github.com/TomF98/torchphysics/blob/main/examples/pinn/exp-function-with-param.ipynb

This approach is also possible for complexer problems, see for example this notebook_.
Where we apply this idea to the heat equation.

.. _notebook: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/heat-equation.ipynb

Inverse heat equation
=====================
For an inverse problem we consider the heat equation:

.. math::

   \begin{align*}
   \text{div}(D(x)\nabla u(x, t)) &= \partial_t u(x, t), \text{ in } \Omega \times [0, 5]\\
   u(x, 0) &= 100\sin(\tfrac{\pi}{10}x_1)\sin(\tfrac{\pi}{10}x_2),  \text{ in } \Omega \\
   u(x, t) &= 0, \text{ on } \partial \Omega \times [0, 5]
   \end{align*}  

with :math:`\Omega = [0, 10] \times [0, 10]`. Here :math:`D` can either be 
a constant value or function itself. Here we start with some data :math:`u(t_i, x_i)`
and want to find the corresponding :math:`D`.

The aim of the following two examples is to show how one can implement this in
TorchPhysics:

- `Constant-D-notebook`_
- `Space-dependent-D-notebook`_

.. _`Constant-D-notebook`: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/inverse-heat-equation.ipynb
.. _`Space-dependent-D-notebook`: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/inverse-heat-equation-D-function.ipynb

Heat equation on moving domain
==============================
To demonstrate how easy one can create a time (or parameter) dependent domain, 
we consider the PDE:

.. math::

   \begin{align*}
      \partial_t u - D\Delta u &= 0,  \text{ in } \Omega \times [0, T] \\
      u(\cdot, 0) &= 0, \text{ in }\Omega \\
      u &= 0, \text{ on } \Gamma_\text{out}  \times [0, T] \\
      \vec{n} \nabla u &= q_\text{in}, \text{ on } \Gamma_\text{in}(t)  \times [0, T]
   \end{align*} 

Where :math:`\Omega` will be a circle with a moving hole and :math:`\Gamma_\text{in}(t)`
the boundary of the hole. The animation on the main page belongs to the
solution of this problem. 

Link to the notebook: `moving-domain-notebook`_ 

.. _`moving-domain-notebook`: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/moving-heat-equation.ipynb


Using hard constrains
=====================
For some problems, it is advantageous to build some prior knowledge into the used network architecture 
(e.g. scaling the network output or fixing the values on the boundary). This can easily be achieved 
in TorchPhysics and is demonstrated in this `hard-constrains-notebook`_. There we consider the system:

.. math::

   \begin{align*}
      \partial_y u(x,y) &= \frac{u(x,y)}{y}, \text{ in } [0, 1] \times [0, 1] \\
      u_1(x, 0) &= 0 , \text{ for } x \in [0, 1] \\
      u_2(x, 1) &= \sin(20\pi*x) , \text{ for } x \in [0, 1] \\
      \vec{n} \nabla u(x, y) &= 0 , \text{ for } x \in \{0, 1\}, y \in \{0, 1\}\\
   \end{align*} 

where the high frequency is problematic for the usual PINN-approach.

.. _`hard-constrains-notebook`: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/hard-constrains.ipynb


Interface jump
==============
For an example where we want to solve a problem with a discontinuous solution, 
we study, for :math:`\Omega = (0, 1) \times (0, 1)` and :math:`\Gamma` the line form
:math:`(0.5, 0)` to :math:`(0.5, 1)`, the PDE:

.. math::

   \begin{align*}
      \Delta u_i &= 0, \text{ in } \Omega \\
      u_1(0, y) &= 0 , \text{ for } y \in [0, 1] \\
      u_2(1, y) &= 2 , \text{ for } y \in [0, 1] \\
      \vec{n} \nabla u_i(x, y) &= 0 , \text{ for } x \in [0, 1], y \in \{0, 1\}\\
      \vec{n} \nabla u_i &= u_2 - u_1, \text{ for } x \in \Gamma
   \end{align*}

with :math:`i = 1, 2` and the solution :math:`u=(u_1, u_2)`, 
split up into left and right part.

For this problem we need two networks, since one alone can, in general, not 
approximate the jump of the solution. Therefore, this example focus on the training of two neural
networks on disjoint domains, coupled over the interface.

Link to the notebook: `jump-notebook`_ 

.. _`jump-notebook`: https://github.com/boschresearch/torchphysics/blob/main/examples/pinn/interface-jump.ipynb