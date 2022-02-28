========
Examples
========
Here we want to mention and link some examples and applications 
that, in our opinion, nicely present the functionalities of TorchPhysics
and the PINN idea.

All currently implemented examples can be found under the `examples-folder`_.

.. _`examples-folder`: https://github.com/boschresearch/torchphysics/tree/main/examples

Poisson problem
===============
The simplest application is the forward solution of a Poisson equation, of the form: 

.. math::

   \begin{align}
   \Delta u &= 4.25\pi^2 u \text{ in } \Omega = [0, 1] \times [0, 1] \\
   u &= \sin(\frac{\pi}{2} x_1)\cos(2\pi x_2) \text{ on } \partial \Omega
   \end{align}

This problem is part of the tutorial and only mentiond for completeness. 
The corresponding implementation can be found here_.

.. _here : tutorial/solve_pde.html

Inverse heat equation
=====================


Heat equation on moving domain
==============================


Interface jump
==============