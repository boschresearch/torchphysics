=========
Changelog
=========
All notable changes to this project will be documented in this file.


Version 1.0
===========
First official release of TorchPhysics on PyPI.

Version 1.0.1
=============
    - Updated documentation and error messages
    - Simplyfied creation/definition of DeepONets
    - Add more evalution types for the DeepONet

Version 1.0.2
=============
    - Test for python versions up to 3.10


Version 1.1.0
=============
    - Rework of operator learning functionalities:
        - Simplification of function spaces
        - Restructuring of function set class (distinguishing between continuous and discrete functions)
        - Addition of function samplers
        - Generalization of operator training conditions
    - Rework of DeepONet and FNO implementation, to make them compatible with the above changes
    - Added PCANN and corresponding data analysis tools (PCA)
    - Simplified saving and loading of neural networks
    - Added discrete differential operators
    - Updated operator learning examples
