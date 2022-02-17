==============
TorchPhysics
==============

TorchPhysics is a Python library of deep learning methods for solving differential equations.
You can use TorchPhysics to: 

- solve ordinary and partial differential equations via physics-informed neural networks (PINN) [1]_ 
  or the Deep Ritz method [2]_
- train a neural network to approximate solutions for different parameters
- solve inverse problems and interpolate external data via the above methods

TorchPhysics is build upon the machine learning library PyTorch_. 

.. _PyTorch: https://pytorch.org/

Features
========
The Goal of this library is to create a basic framework that can be used in many
different applications and with different deep learning methods.
To this end, TorchPhysics aims at a:

- modular and expandable structure
- easy to understand code and clean documentation
- intuitive and compact way to transfer the mathematical problem into code
- reliable and well tested code basis 

Some build in features are:

- mesh free domain generation. With pre implemented domain types: 
  *Point, Interval, Parallelogram, Circle, Triangle and Sphere*
- loading external created objects, thanks to a soft dependency on Trimesh_  
  and Shapely_
- creating complex domains with the boolean operators *Union*, *Cut* and *Intersection* 
  and higher dimensional objects over the Cartesian product
- allowing interdependence of different domains, e.g. creating moving domains
- different point sampling methods for every domain:
  *RandomUniform, Grid, Gaussian, Latin hypercube, Adaptive* and some more for specific domains
- different operators to easily define a differential equation
- pre implemented fully connected neural network and easy implementation
  of additional model structures 
- sequentially or parallel evaluation/training of different neural networks
- normalization layers and adaptive weights [3]_ to speed up the training process
- powerful and versatile training thanks to `PyTorch Lightning`_
  
  - many options for optimizers and learning rate control
  - monitoring the loss of individual conditions while training 


.. _Trimesh: https://github.com/mikedh/trimesh
.. _Shapely: https://github.com/shapely/shapely
.. _`PyTorch Lightning`: https://www.pytorchlightning.ai/


Getting Started
===============
To learn the functionality and usage of TorchPhysics we recommend
to have a look at the following sections:

- `Tutorial: Understanding the structure of TorchPhysics`_
- `Examples: Different applications with detailed explanations`_
- Documentation_

.. _`Tutorial: Understanding the structure of TorchPhysics`: https://torchphysics.readthedocs.io/en/latest/tutorial/tutorial_start.html
.. _`Examples: Different applications with detailed explanations`: https://github.com/boschresearch/torchphysics/tree/main/examples
.. _Documentation: https://torchphysics.readthedocs.io/en/latest/api/modules.html


Installation
============
TorchPhysics can be installed by using:

.. code-block:: python

  pip install git+https://github.com/boschresearch/torchphysics

If you want to change or add something to the code. You should first copy the repository and install
it locally:

.. code-block:: python

  git clone https://github.com/boschresearch/torchphysics 
  pip install .


About
=====
TorchPhysics was originally developed by Nick Heilenkötter and Tom Freudenberg, 
as part of a `seminar project`_ at the `University of Bremen`_, in cooperation with the `Robert Bosch GmbH`_. 
Special thanks belong to Felix Hildebrand, Uwe Iben, Daniel Christopher Kreuter and Johannes Mueller,
at the Robert Bosch GmbH, for support and supervision while creating this library.

.. _`seminar project`: http://www.math.uni-bremen.de/zetem/cms/detail.php?template=modellierungsseminar
.. _`University of Bremen`: https://www.uni-bremen.de/en/
.. _`Robert Bosch GmbH`: https://www.bosch.de/en/

License
=======
TorchPhysics uses a Apache License, see the LICENSE_ file.

.. _LICENSE: https://github.com/boschresearch/torchphysics/blob/main/LICENSE.txt


Bibliography
============
.. [1] Raissi, Perdikaris und Karniadakis, “Physics-informed neuralnetworks: A deep learning framework for solving forward and inverseproblems involving nonlinear partial differential equations”, 2019.
.. [2] E and Yu, "The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems", 2017
.. [3] McClenny und Braga-Neto, “Self-Adaptive Physics-Informed NeuralNetworks using a Soft Attention Mechanism”, 2020