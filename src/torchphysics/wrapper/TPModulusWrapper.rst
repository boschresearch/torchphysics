===============================
TorchPhysics to Modulus Wrapper
===============================

This folder contains a wrapper module for TorchPhysics to `NVIDIA Modulus`_.
This module serves as a bridge between the two frameworks and allows users to train TorchPhysics models with the Modulus training framework with minimal changes to their existing code. 

Both libraries are based on Pytorch_, but instead of `PyTorch Lightning`_, Modulus uses its own *Distributedmanager*, `TorchScript`_ with Pytorch JIT backend and CUDA graphs as a further framework to optimize and accelerate the training process, especially on NVIDIA GPUs.

Both libraries use ``torch.nn.Module`` as the base class for the model definition, so that any model architecture of Modulus can easily be used as a TorchPhysics model, which is one of the main purposes of this wrapper module.
Modulus offers a wide range of model architectures, including various types of Fourier neural networks.

As a second purpose, the wrapper module can be used to automatically convert a TorchPhysics problem to Modulus and perform the training in Modulus to benefit from the optimizable Modulus training framework.

In both libraries, geometries are defined for the spatial domain of the problem (PDE or ODE), but the Modulus geometry provides a signed distance function (SDF) that calculates the distance of any arbitrary point to the boundary of the geometry.
This makes it possible to weight the loss function of the problem with the distance to the boundary, which is recommended in Modulus, e.g. for sharp gradients near the boundary, as it can increase convergence speed and improve accuracy.

The wrapper module generally provides spatial loss weighting and the loss balancing algorithms of Modulus, e.g. GradNorm or Learning Rate Annealing.

Each of the two main classes of the wrapper module can be used independently, allowing the user to choose whether to use only the conversion of the problem or only the model architecture, or both.

Variable conventions
====================
The spatial and temporal variables in Modulus are always defined as 'x', 'y', 'z' and 't', so TorchPhysics models can only be converted if the spatial variable is defined as 'x', which can be 1D, 2D or 3D, or 'x', 'y' or 'x', 'y','z' and the temporal variable as 't'.

Contents
========
The wrapper module consists of two main classes:

* ``TPModulusWrapper``: can be used to convert a TorchPhysics problem into Modulus and to perform training in Modulus. ``help(TPModulusWrapper)`` provides a detailed description of the class and its parameters.

* ``ModulusArchitectureWrapper``: can be used to choose any implemented Modulus architecture as TorchPhysics model. ``help(ModulusArchitectureWrapper)`` provides a detailed description of the class and its parameters.

Usage
=====
To use the wrapper module to run a TorchPhysics model in Modulus, you can add the following line to your existing TorchPhysics code after the definition
of the ``trainer`` and the ``solver`` object, replacing the ``trainer.fit(solver)`` call: 

.. code-block:: python
    
    torchphysics.wrapper.TPModulusWrapper(trainer,solver).train()


To use one of the Modulus model architectures, you can add the following line to your existing TorchPhysics code and replace the model definition,
e.g. if you want to use the Modulus Fourier architecture as a TorchPhysics model:

.. code-block:: python
    
    model=torchphysics.wrapper.ModulusArchitectureWrapper(input_space=X*T, output_space=U,arch_name='fourier',frequencies = ['axis',[0,1,2]])


Installation
============
The wrapper module requires a working installation of TorchPhysics and Modulus Symbolic (Sym), which is a framework providing algorithms
and utilities to be used with Modulus core for physics informed model training.

The installation of Modulus Sym is documented here: `NVIDIA Modulus Github Repository`_

We recommend to create a new conda environment and to first install NVIDIA Modulus with the following command:

.. code-block:: python
    
   pip install nvidia-modulus.sym

Then you can install TorchPhysics as described in the `TorchPhysics documentation`_.

As Modulus Sym uses TorchScript_ by default to compile the model, it is important to have the correct version of PyTorch_ installed. The current Modulus Sym version requires 2.1.0a0+4136153.
If a different version is installed, a warning will be raised when starting the training and the use of TorchScript is disabled.
The version can be determined by the command

.. code-block:: python

    torch.__version__

To circumvent disabling of TorchScript you can edit the file /modulus/sym/constants.py in your python site packages installation path, and change the constant JIT_PYTORCH_VERSION = "2.1.0a0+4136153" to the version you have installed, e.g. "2.4.0+cu121".

.. _`PyTorch Lightning`: https://www.pytorchlightning.ai/
.. _`NVIDIA Modulus`: https://developer.nvidia.com/modulus
.. _`NVIDIA Modulus Github Repository`: https://github.com/NVIDIA/modulus-sym/tree/main
.. _PyTorch: https://pytorch.org/
.. _TorchScript: https://pytorch.org/docs/stable/jit.html
.. _`TorchPhysics documentation`: https://github.com/boschresearch/torchphysics/blob/main/README.rst



Testing
=======
As the wrapper module needs additional installation steps and can not be used without Modulus, it is excluded from the automatic testing with pytest. To test the functionality of the wrapper, there are example notebooks in the folder examples/wrapper and tests in src/torchphysics/wrapper/tests that can be manually invoked by the command (requires the installation of pytest and pytest-cov):

.. code-block:: python

    pytest src/torchphysics/wrapper/tests



Some notes
==========
* The loss definition in Modulus is based on Monte Carlo integration and therefore the loss is scaled proportional to the corresponding area, i.e. it is usually different from the loss in TorchPhysics, where the loss is the mean value.
* Currently, ``stl``-file support in Modulus is only available for Docker installation, so ``shapely`` and ``Trimesh`` geometries in TorchPhysics can not be converted.
* Cross product domains are generally not supported in Modulus, so must be automatically converted by the wrapper to existing primary geometries, so not all combinations of domain operations are allowed, e.g. product domains only from the union of 1D or 0D domains and no further rotation and translation is allowed (must be done with the entire product).
* Physics-Informed Deep Operator Networks (PIDOns) are currently not supported in the wrapper. 
* Fourier Neural Operators (FNOs) are currently not supported in the wrapper, but an FNO framework is currently being developed in TorchPhysics.
* Samplers other than random uniformn and Halton sequence are not supported in Modulus.
* The imposition of exact boundary conditions using hard constraints with Approximate Distance Functions (ADFs) is not yet supported in TorchPhysics.
* The Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer can be used in Modulus by setting the maximum step size (``max_steps``) to 1 (one single optimization step), but using the maximum number of iterations per optimization step (``max_iter``) as the number of iterations instead. This is very slow, so it is recommended to use Adam instead. In TorchPhysics, ``max_iter`` is decreased and many optimization steps are performed.
* If the combination of the Adam and L-BFGS optimizers is used, then loading the L-BFGS optimizer checkpoint file (optim_checkpoint.0.pth) will result in an error regarding ``max_iter`` as Adam does not use ``max_iter``. This is a known issue for Modulus support and it is recommended to delete or rename the optim_checkpoint.0.pth file. Then it works, but Tensorboard cannot display the loss history correctly!
* If several losses with the same name of the objective variable are used, the losses are summarized in Tensorboard, e.g. initial condition for T and Dirichlet condition for T, then there is only one loss (sum) for T.
* In general, all TorchPhysics callbacks are supported, but for the ``WeightSaveCallback``  the check for minimial loss (parameter ``check_interval``) is not supported by the wrapper, only initial and final model states are saved.
* Modulus automatically provides Tensorboard logging of the losses. The corresponding logging folder is ``outputs`` by default, but can be set by the user with the parameter ``outputdir_name``.
* Modulus automatically provides ``.vtp``-files containing data computed on the collocation points of the conditions that can be found in subfolders of the output directory. These files can be viewed using visualization tools like Paraview.