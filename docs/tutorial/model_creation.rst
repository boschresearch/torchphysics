===================================
Definition of Models and Parameters
===================================
The models are the neural networks that should be trained to fulfill user-defined conditions.
What basics network structures are implemented, can be found under the `model-docs`_.
All networks can found under ``torchphysics.models``. For example, a simple fully 
connected neural network can be created with:

.. _`model-docs`: https://torchphysics.readthedocs.io/en/latest/api/torchphysics.models.html

.. code-block:: python

  import torchphysics as tp
  X = tp.spaces.R2('x') # for the input dimension of the network
  U = tp.spaces.R1('u') # defines the output dimension of the networks
  model = tp.models.FCN(input_space=X, output_space=U, hidden=(30,30,30))

Parameters that should be learned in a inverse problem can also be easily defined:

.. code-block:: python 

  D_space = tp.spaces.R1('D') # parameters need there own space of corresponding dimension
  D = tp.models.Parameter(init=0.0, space=D) # here you have to pass a fitting inital guess 

That are all the basics to the creation of the networks and parameters, they could now be used in a 
condition to then start the training.

Sequential and Parallel evaluation
----------------------------------
For complex problems, there may be the need for more complex network structures. For this
two additional functionalities are implemented:

- **Sequential**: The sequential evaluation of different neural networks. For example a 
  normalization layer, that scales all input values in the range [-1, 1], can be
  applied this way.
- **Parallel**: The parallel evaluation of different neural networks. For example, if the
  solution of the PDE consists of two different functions, e.g. velocity :math:`v` and 
  pressure :math:`p`, or locally different networks should be applied.

In the following the application of the normalization:

.. code-block:: python

  # we first need a domain to know how to scale the points
  T = tp.domains.Triangle(X, origin=[0, 0], corner_1=[1, 0], corner_2=[2.0, 0])
  normal_layer = tp.models.NormalizationLayer(T) # pass in the domain
  seq_model = tp.models.Sequential(normal_layer, model) # the evaluation will be from left to right


Using your own neural network
-----------------------------
Since the models are build upon PyTorch, a custom network can be easily implemented. 
But there are two points that have to be remembered for this:

1) The network has to have an ``input_space`` and ``output_space``. They should be set in the
   initialization of the custom neural network. But this applies naturally if you extend the
   implemented parent class ``Model``.
2) In the forward call a ``Point``-object has to be returned. These are the underlying custom
   tensors of TorchPhysics, like explained in the `space tutorial`_. For this you just have 
   do the following:

   .. code-block:: python

    from torchphysics.problem.spaces import Points
    class YourModel(...):
        ...
        def forward(points):
            # here your computations....
            return Points(your_model_output, self.output_space)

.. _`space tutorial`: tutorial_spaces_and_points.html

These are all the basics to creation of neural networks and parameters, next up are either 
the **Conditions** or you can have a look at the **utils**, here_ you can go back to the main site.

.. _here: tutorial_start.html