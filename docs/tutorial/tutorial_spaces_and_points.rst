=================================
Spaces and points in TorchPhysics
=================================
In this tutorial, we will cover the starting point for every PDE setting: The involved
spaces which define the names and dimensionlities of all variables.

Spaces
------
The class **Space** itself is quite lazy and basically consists of a counter collecting
dimensionlities of space variables. It's purpose is to define variable names that can later
be used, e.g. in user-defined functions. They therefore appear in several parts of TorchPhysics,
for example in the definition of domains or models.

A ``Space`` is best defined as the cartesian product of its subclasses ``R1``, ``R2`` or ``R3``,
which build the dimensions that belong to a single variable. For example, if you want to work
with 2 spatial dimensions called ``x`` and a one-dimensional time variable ``t``, you could
define your spaces as follows:

.. code-block:: python

   X = tp.spaces.R2('x')
   T = tp.spaces.R1('t')
   G = X*T

A ``Space`` collects all variable dimensions and keeps their order. It has a dimensionality

.. code-block:: python

   >>> G.dim
   3

and one can access its variables as an (unordered) set:

.. code-block:: python

   >>> G.variables
   {'x', 't'}

Furthermore, we can check subspace-relations and whether the space contains a variable using
the ``in``-operator:

.. code-block:: python

   >>> 'x' in G
   True
   >>> X in G
   True
   >>> T*X in G
   True
   >>> 'y' in G
   False

Points
------
The ``Points`` object is another central part of TorchPhysics. It consists of a PyTorch-tensor
collecting a set of points in a ``Space``. It is generated e.g. by the samplers during training
and handed to and from all models as in- and output. However, for standard use-cases, ``Points``
mostly stay behind the scenes, so if you don't need custom behaviour when using TorchPhysics, feel
free to skip this part of the tutorial for now.

``Points`` store data in a tensor with 2-axis, the first corresponding the batch-dimension in a batch
of multiple points. The second axis collects the space dimensionalities.

A set of points can be created by their coordinates:

.. code-block:: python

   x = torch.randn(10, 2)
   t = torch.randn(10, 1)
   points = tp.spaces.Points.from_coordinates({'x': x, 't': t})

All ``Points`` have a space and therefore also a dimensionality and a variable set:

.. code-block:: python

   >>> points.space
   Space({'x': 2, 't': 1})
   >>> points.dim
   3
We can access the contents of a ``Points`` object in a single tensor or with the corresponding coordinate
dict using ``.as_tensor`` or ``.coordinates`` attribues. ``Points`` also support most torch functions that
work on tensors and support slicing via keys along the ordered variable axis, regarding the last key in slicing
(similar to NumPy or PyTorch-behaviour):

.. code-block:: python

   >>> points[1:3, 'x':'t']
   Points:
   {'x': tensor([[-1.0599,  0.7874],
                 [ 0.1690,  1.3649]])}
   >>> points[1:3, 't':]
   Points:
   {'t': tensor([[-0.8097],
                 [ 0.1553]])}


You should now have a basic understanding of spaces and points in TorchPhysics. For more details
on specific features, also take a look at the generated docs.
The next step in this tutorial are `basics on domains`_.

.. _`basics on domains`: tutorial_domain_basics.html