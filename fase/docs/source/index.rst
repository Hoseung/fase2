.. FASE documentation master file, created by
   sphinx-quickstart on Sun Feb 27 09:28:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FASE's documentation!
================================

This package is a Python interface to Homomorphic Encryption(HE) libraries -- 
`HEAAN <https://github.com/snucrypto/HEAAN>`_, 
`MS-SEAL <https://github.com/microsoft/SEAL>`_, 
and `PALISADE <https://gitlab.com/palisade/palisade-release>`_ for ease of use.
It also includes custom versions of HEAAN library enabling FPGA and CUDA acceleration.
Underlying HE libraries are all written in C++ and are bound to Python via `pybind11 <https://github.com/pybind/pybind11) or [Boost.Python](https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/index.html>`_

.. toctree::
   :maxdepth: 3
   :caption: Contents

   Getting Started <GettingStarted>
   modules <modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
