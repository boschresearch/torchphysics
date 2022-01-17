=====================================================
Tutorial: Understanding the structure of TorchPhysics
=====================================================
In this tutorial, you will learn how the different components of TorchPhysics
work and at the end will be able to transform a simple differential equation into the
corresponding problem in TorchPhysics.

We start by explaining the basic structure of this library. Then we will go over each 
important part in their own tutorial and in the end bring everything together to
solve a PDE.

The structure of TorchPhysics is best illustrated by the following graph:

.. image:: pictures/torchphysics_structure.png
  :width: 500
  :align: center
  :alt: Graph of the work flow in TorchPhysics


Spaces 
  Define the dimension of the used variables, parameters and model outputs. This is the
  starting point for all problems and shown in the `spaces and points tutorial`_.
  There it will be also explained, with what kind of objects the library stores and creates
  points.

Domains 
  Handle the creation of the underlying geometry of the problem. See:
  
  - `Domain basics`_ to learn everything about the definition and functionalities 
  - `Polygons and external objects`_ to create 2D or 3D polygons and import external files

PointSampler
  Control the creation of sampling points for the trainings/validation process. The usage
  is explained in the `PointSampler tutorial`_.

Models/Parameters
  Implement different neural network structures and trainable parameters. 
  How to define a model is shown in the `model creation tutorial`_. 

Conditions 
  Combine the created *Domains*, *PointSampler* and *Models* to apply the conditions
  induced by the differential equation. See `condition tutorial`_ on how to create different
  kinds of conditions for all parts of the problem.

Utils
  Implement a variety of helper functions to make the definition and evaluation of 
  problems easier. To get an overview of all methods, see the docs_. Two parts that will
  be shown more detailed, are:

  - Here_ the usage of the pre implemented differential operators
  - `Creating plots`_ of the trained solutions.

Solver
  Handles the training of the defined model, by applying the previously created conditions.
  The usage of the solver, will be shown in a complete probleme, where all the above points
  are used. This is shown in `solving a simple PDE`_.

These are all the basics of TorchPhysics. You should now have a rough understanding of the 
structure of this library. Some additional applications (inverse problems, training input params, ...)
can be found under the `example-folder`_

.. _`spaces and points tutorial`: tutorial_spaces_and_points.rst
.. _`Domain basics`: tutorial_domain_basics.rst
.. _`Polygons and external objects`: external_domains.rst
.. _`PointSampler tutorial`: sampler_tutorial.rst
.. _`model creation tutorial`: model_creation.rst
.. _`condition tutorial`: condition_tutorial.rst
.. _docs: missing
.. _Here: differentialoperators.rst
.. _`Creating plots`: plotting.rst
.. _`solving a simple PDE`: solve_pde.rst
.. _`example-folder`: https://github.com/boschresearch/torchphysics/tree/main/examples