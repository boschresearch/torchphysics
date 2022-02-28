============================
Use of Differentialoperators
============================
To learn a solution of a differential equation, one needs to compute different 
derivatives of the neural network.
To make the implementation of a given ODE/PDE easier, different operators are already 
implemented. They can be found under ``torchphysics.utils.differentialoperators``.
Under the hood, all operators use the ``autograd`` functionallity of PyTorch.
For example the following operators are implemented:

- For scalar outputs:

  - ``grad``, to compute the gradient :math:`\nabla u` 
  - ``laplacian``, to compute the laplace operator :math:`\Delta u`
  - ``partial``, to compute a partial derivative :math:`\partial_x u`
  - ``normalderivatives``, to compute the normal derivative :math:`\vec{n} \cdot \nabla u`

- For vector outputs:

  - ``div``, to compute the divergence :math:`\text{div}(u)`  or :math:`\nabla \cdot u` 
  - ``rot``, to compute the rotation/curl of a vector field :math:`\nabla \times u`
  - ``jac``, to compute the jacobian matrix

All operators can handle the computation on a whole batch of data.
Of course, the operators for scalar outputs can also be used for vectorial outputs, if one output 
entry is specified. E.g. :math:`u: \mathbb{R}^3 \to \mathbb{R}^3` then do 
``laplacian`` (:math:`u[:, 0], x`), to get the laplacian of the first entry.
The newest version of all implemented operators can be found under the docs_.

.. _docs: https://torchphysics.readthedocs.io/en/latest/api/torchphysics.utils.html

Since ``autograd`` is used, the differential operators can work with neural networks and functions
that use PyTorch-Tensors. It follow a short example of the usage:

.. code-block:: python 

  import torch
  # Define some example function:
  def f(x, t):
    return torch.sin(t * x[:, :1]) + x[:, 1:]**2

  # Define some points where to evaluate the function
  x = torch.tensor([[1.0, 1.0], [0, 1], [1, 0]], requires_grad=True) 
  t = torch.tensor([[1], [0], [2.0]], requires_grad=True)
  # requires_grad=True is needed, so PyTorch knows to create a backwards graph.
  # These tensors could be seen as a batch with three data points.

The important part for the implemented operators and ``autograd`` in general, is that the output
of the function evaluated at the points is needed, not the function itself. This has the advantage 
that one has to only evaluate the function once and then can create arbitrary derivatives.

.. code-block:: python 

  # Therefore comput now the outputs:
  out = f(x, t)

Let us compute the gradient and laplacian:

.. code-block:: python 

  import torchphysics as tp
  # gradient and laplacian w.r.t. x:
  grad_x = tp.utils.grad(out, x)
  laplace_x = tp.utils.laplacian(out, x)
  # gradient and laplacian w.r.t. t:
  grad_t = tp.utils.grad(out, t) # equal to the first derivative
  laplace_t = tp.utils.laplacian(out, t) # equal to the second derivative

What is also possible is the computation of derivative w.r.t. different variables. For
this, one just has to pass all variables, for which the derivative has to be computed, to the method.
E.g. for :math:`\partial_{x_1}^2f + \partial_{x_2}^2f + \partial_t^2f` one can use:

.. code-block:: python 

  laplace_t = tp.utils.laplacian(out, x, t) # <- here both variables

All other operators work in the same way. Here_ you can go back to the main tutorial page.

.. _Here: tutorial_start.html