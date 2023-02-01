.. Error Estimation for FEM Using Neural Networks documentation master file, created by
   sphinx-quickstart on Wed Jan 25 14:29:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Error Estimation for FEM Using Neural Networks's documentation!
==========================================================================

The goal of this project is to develop an error estimator for the Finite Element Method (FEM). The error estimator is a neural network
that uses the properties a FEM as input and outputs a local error estimate. We have incorporated the estimator in an Adaptive Mesh
Refinement (AMR) pipeline to test its efficacy. The error estimator (neural network) can be trained by user-provided data and be used 
alternatively in the AMR pipeline, with restrictions on the input and output vector dimensions.  

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   set_up.rst

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation
   
   amr.rst

.. toctree::
   :maxdepth: 2
   :caption: Utilities 
   
   plot_utils.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
