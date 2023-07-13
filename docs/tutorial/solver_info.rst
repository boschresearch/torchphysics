==================================
Info about the Solver and Training
==================================
For solving a differential equation and training the neural network, we use the 
library Pytorch Lightning. This lets us easily handles the training and validation 
loops and take care of the data loading/creation. 

To see the solver class in action, we refer to the `beginning example`_ of the tutorial. 
There, the behavior of the TorchPhysics ``Solver`` is explained and shown. 
Here we rather want to mention some more details for the trainings process.

In general, most of the capabilities of Pytorch Lightning can also be used inside 
TorchPhysics. All possiblities can be checked in the `Lightning documentation`_.

.. _`beginning example`: solve_pde.html
.. _`Lightning documentation`: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

Given some TorchPhysics ``Solver``, that already contains all information of our problem, 
the basic trainer is defined as follows:

.. code-block:: python
  
  import pytorch_lightning as pl

  trainer = pl.Trainer(gpus=1, max_steps=4000, check_val_every_n_epoch=10)
  trainer.fit(solver) # start training

Some important keywords are:

- **gpus**: The number of GPUs that should be used. If only one or more CPUs are available
  set ```gpus=None``. Depending on the operating system, the GPUs may have to be further
  specified beforehand via ``os.environ`` or other ways. There are more different possiblities to 
  specify the used device, see the above mentionde documentation.
- **max_steps**: The maximum number of training iterations. In each iteration
  all defined training conditions will be evaluated 
  (e.g. points sampled, model output computed, residuals evaluated, etc.) and a 
  gradient descent step will be made.
- **check_val_every_n_epoch**: Defines how often the validation data should be checked. 

Logging Data 
------------
Using the above definition for the ``trainer`` will lead while training to the creation of a folder named
*lightning_logs* inside the current directory. Inside this folder, data about the current training
process is saved. In TorchPhysics this data includes the loss of each training condition, over the 
course of all iterations, and the total loss. 
(Here it is important to give each condition a unique ``name``, or the data may be overwritten) 

While the training is running, or afterwards, the loss can be monitored inside TensorBoard, if installed. 
For this, use inside a terminal:

.. code-block:: console
  
  tensorboard --logdir path_to_log_folder

Which will open a new window inside your browser to visualize the data. Some important keywords of 
the ``pl.Trainer`` regarding logging are:

- **logger**: A ``TensorBoardLogger`` that defines the saving behavior. As default, a Logger with
the above mentioned aspects is used. Setting ``logger=False`` will disable logging.
- **log_every_n_steps**: How often data should be saved.

Callbacks
---------
Callbacks are an additional versatile option to monitor the training or apply custom logic, while 
training. For example, via Callbacks the learned solution can be plotted to TensorBoard or after
every few steps the trained network can be saved. Different callbacks can be created via
``pytorch_lightning.callbacks``. Already implemented callbacks in TorchPhysics are found under
the ``torchphysics.utils`` section. A list of all created callbacks can then be passed to the trainer 
with the ``callbacks`` keyword.