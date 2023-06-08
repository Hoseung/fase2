Getting Started
***************

Installation Guide
==================

As the package includes three different HE libraries, installation takes multiple steps. 
To keep things simple, we recommend installing the package in a Python virtual environment. 
At high level, the installation process consists of three steps: 
1. Setup Python environment
2. Install each HE libraries
3. Install the FASE Python package

1. Conda Environment
--------------------
The list of prerequisite pacakges is given in requirements.yml. 
You can create a conda environment and install necessary packages as:

.. code-block:: bash

   $ conda env create  --file requirements.yml


2. Building Homomorphic Encryption Libraries
--------------------------------------------
All three HE libraries either strictly or optionally depends on the Number Theory Library `NTL <https://libntl.org/download.html>`_,
which in turn depends on `GMP <https://ftp.gnu.org/gnu/gmp/>`_. 
So, we install GMP and NTL before HE libraries. 

GMP
+++

.. code-block:: bash

    cd gmp-6.2.1
    ./configure SHARED=on 
    make
    make check  # GMP deveolopers strongly recommned do this
    (sudo) make install 

.. note:: 
    If you encounter version number mismatch error message like
    ``> GMP version check (6.2.1/6.2.0)   
    > *** version number mismatch: inconsistency between gmp.h and libgmp``  
    chances are 

during building NTL library, try installing GMP library system-wide via apt instead.

.. code-block:: bash

    sudo apt install libgmp3-dev 
    sudo ldconfig

NTL
+++
We have tested NTL 11.5.1. Add flags to `Configure` as follows.

.. code-block:: bash

    cd NTL/src
    ./configure SHARED=on NTL_GMP_LIP=on
    make
    make check  # optional
    (sudo) make install


Next, we build three different HE libraries. 
You can simply run the ``./build.sh`` script at the root directory of the pacakge.
But here we provide a detailed explanation in case something need to be adopted to your system.

1. Buildling HEAAN
------------------

The package contains source files of the mainline HEAAN, cuda-accelerated 
0. You can find the HEAAN source code under ./HEAAN/
1. It requires essential development tools such as C++ compiler and Cmake. Install them via apt: ``$ sudo apt install build-essential cmake``
We have tested under conda Env and Python 3.9

GMP 6.2.1 worked fine for us.


Now you are ready to build HEAAN. (need to be compiled with -fPIC)

```bash
cd HEAAN/lib
#    # Make sure in `src/subdir.mk`, there's the `-fPIC` option in g++ call (Line 59)
make 
# And run the test example1
cd run
make
./TestHEAAN bootstrapping
```

Building SEAL
-------------

Provided that build-essential, cmake, and python are available, building SEAL library is simple.

SEAL requires a cmake of more recent version than what is available on Ubuntu18.04.
Within conda environment, install newer cmak with pip as:
`pip install --upgrade cmake`

```bash
cd SEAL
cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
cmake --build build
```

Now, back in the root dir, run:

```bash
python3 setup.py build_ext -i
```

to install *Fase*.

You will see `HEANN.cpython.....so` and `seal.cpython.....so` among others.
