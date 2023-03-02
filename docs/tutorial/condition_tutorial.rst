==========
Conditions
==========
The **Conditions** are the central concept of TorchPhysics. They transform the conditions of
the underlying differential equation into the trainings conditions for the neural network.

Many kinds of conditions are pre implemented, they can be found in
the docs_. Depending on the used type, the condition get different inputs. The five arguments 
that almost all conditions need, are:

.. _docs: https://torchphysics.readthedocs.io/en/latest/api/torchphysics.problem.conditions.html

module
    Your model that should be trained, with the condition you are creating. The definition
    of different kind of model structures were part of the `previous tutorial`_.

sampler 
    The sampler that will provide the points on which the condition should be applied.
    These were part of the `sampler tutorial`_.

residual_fn
    A function that computes the residual corresponding to the condition, by using the
    input and output of the model. How they should be defined is the focus of the following
    part.

name 
    A name for the condition. Only useful if logging in the trainings process is applied, 
    to then know which loss corresponds to which condition.

weight
    A additional weight, that will be multiplied with the loss in the training.


From the output of the **residual_fn** the loss will be computed. The default way of doing this is the 
mean squared error. This can be customized by changing the corresponding ``error_fn`` and ``reduce_fn`` 
of the condition.

If even more customization is needed, one can easily extend the main ``Condition``-class, which is essentially
just a ``torch.nn.Module``.

All conditions can also be applied for validation or testing.

.. _`previous tutorial`: model_creation.html
.. _`sampler tutorial`: sampler_tutorial.html


Defining a Residual-Function
----------------------------
Now we focus on the structure of the functions that compute the residual in the trainings process.
For this, let us assume our problem has the input variables ``t``, ``x`` and ``D`` and the output
``u``. How one would implement the corresponding spaces and domains was part of the previous
tutorials and is now not really important. Here we only need the names of the variables.

First, let's say we have a boundary condition of the form:

.. math::

   u(t, x) = \sin(t \cdot x_1) \cdot \sin(x_2) \text{ on } \partial \Omega

There we see the ``D`` has **no** influence in this condition and is for now not needed. We only
need the output ``u`` and the input ``t`` and ``x``. Therefore the corresponding residual would be:

.. code-block:: python

    def boundary_residual(u, t, x):
        return u - torch.sin(t * x[:, :1])*torch.sin(t * x[:, 1:])

Here many **important** things are happening:

1) The inputs of the function ``boundary_residual`` only corresponds to the variables required. 
   The needed variables will then internally be correctly passed to the method. Here it is important
   to use the correct names that were used in the ``Spaces``.
2) We have the input ``u``, that is already the output of our neural network. So we **do not** need
   to call ``u(t, x)`` first, this also happens beforehand.
3) The input variables will be always in the shape of: [batch-dimension, space-dimension].
   If we assume ``x`` is two-dimensional the value ``x[:, :1]`` corresponds to the entries
   of the first axis, while ``x[:, 1:]`` is the second axis.
   One could also write something like ``t[:, :1]``, but this is equal to ``t``.
   Maybe now the question arises, if one could also use ``x[:, 0]`` instead of ``x[:, :1]``.
   Both expressions return the first axis of our two-dimensional input points. **But** 
   there is one important difference in the output, the final **shape**. The use of
   ``x[:, 0]`` leads to an output of the shape: [batch-dimension] (essentially just 
   a list/tensor with the values of the first axis), while ``x[:, :1]`` has the 
   shape: [batch-dimension, 1]. So ``x[:, :1]`` preserves the shape of the 
   input. This is important to get the correct behavior for different operations like
   addition, multiplication, etc. Therefore, ``x[:, 0]`` should generally not be used
   inside the residuals and can even lead to errors while training. For more info 
   on tensor shapes one should check the documentation of `PyTorch`_.
4) The output at the end has to be a PyTorch tensor.
5) We have to rewrite the residual in such a way, that the right hand side is zero. E.g.
   here we have to bring the sin-function to the other side.

.. _`PyTorch`: https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

The defined method could then be passed to a condition. Let us assume we have already created 
our model and sampler:

.. code-block:: python

    boundary_condition = tp.conditions.PINNCondition(module=model, sampler=bound_sampler, 
                                                     residual_fn=bound_residual)

In the same way, other conditions can be created. As an example, let us assume we have the PDE:

.. math::

   -D\Delta u = \sin(t \cdot x_1) \text{ in } \Omega

Since ``D`` is also input of the model, we have to now pass it to our residual.
For the Laplacian different operators are implemented, see the `operator tutorial`_ for a small example.

.. _`operator tutorial`: differentialoperators.html

.. code-block:: python

    def pde_residual(u, t, x, D):
        lap_u = tp.utils.laplacian(u, x)
        return - D * lap_u - torch.sin(t * x[:, :1])

The same things as before hold.

These are the basics of the ``Condition``-class, next up would be either some **utils** or 
connecting everything to solve a PDE. Here_ you can go back to the main page.

.. _Here: tutorial_start.html