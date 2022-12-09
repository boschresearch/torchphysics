=====================================================
Tutorial: Understanding the structure of TorchPhysics
=====================================================
In this tutorial, you will learn how the different components of TorchPhysics
work and interact. In the end, you will be able to transform a simple differential equation
into the corresponding training setup in TorchPhysics.

We start by explaining the basic structure of this library. Then we will go over each 
important part in its own tutorial. Finally we will bring everything together to
solve a PDE using the PINN-approach.

The structure of TorchPhysics can be illustrated via the following graph:

.. image:: pictures/torchphysics_structure.png
  :width: 500
  :align: center
  :alt: Graph of the work flow in TorchPhysics


Spaces 
  Define the dimension of the used variables, parameters and model outputs. This is the
  starting point for all problems and shown in the `spaces and points tutorial`_.
  The tutorial also covers the way the library stores and creates points.

Domains 
  Handle the creation of the underlying geometry of the problem. See:
  
  - `Domain basics`_ to learn everything about the definition and functionalities 
  - `Polygons and external objects`_ to create 2D or 3D polygons and import external files

PointSampler
  Control the creation of sampling points for the training/validation process. The usage
  is explained in the `PointSampler tutorial`_.

Models/Parameters
  Implement different neural network structures and trainable parameters. 
  How to define a model is shown in the `model creation tutorial`_. 

Conditions 
  Combine the created *Domains*, *PointSampler* and *Models* to apply the conditions
  induced by the differential equation. See `condition tutorial`_ on how to create different
  kinds of conditions for all parts of the problem.

Solver
  Handles the training of the defined model, by applying the previously created conditions.
  The usage of the solver is shown the beginnig example for `solving a simple PDE`_. More details
  of the trainings process are mentioned here_.

.. _here: solver_info.html

Utils
  Implement a variety of helper functions to make the definition and evaluation of 
  problems easier. To get an overview of all methods, see the docs_. Two parts that will
  be shown more detailed, are:

  - The usage of the pre implemented `differential operators`_
  - `Creating plots`_ of the trained solutions.

.. _`spaces and points tutorial`: tutorial_spaces_and_points.html
.. _`Domain basics`: tutorial_domain_basics.html
.. _`Polygons and external objects`: external_domains.html
.. _`PointSampler tutorial`: sampler_tutorial.html
.. _`model creation tutorial`: model_creation.html
.. _`condition tutorial`: condition_tutorial.html
.. _docs: https://torchphysics.readthedocs.io/en/latest/api/torchphysics.utils.html
.. _`differential operators`: differentialoperators.html
.. _`Creating plots`: plotting.html
.. _`solving a simple PDE`: solve_pde.html
.. _`example-folder`: https://github.com/boschresearch/torchphysics/tree/main/examples