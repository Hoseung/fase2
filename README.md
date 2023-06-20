FHE Binder
==============

Built with [pybind11](https://github.com/pybind/pybind11)

This packages packs three different FHE libraries: [HEAAN](https://github.com/snucrypto/HEAAN), [MS-SEAL](https://github.com/microsoft/SEAL), and [PALISADE](https://gitlab.com/palisade/palisade-release)  
Beware these back-end libraries are being actively developed and open to the changes in their API, build process, and functionality.


Conda environment
-----------------
in the root directory of the package:
```bash
$ conda env create  --file requirements.yml
```


Buildling HEAAN
---------------

0. You can find the HEAAN source code under ./HEAAN/
1. It requires, of course, essential development tools such as C++ compiler and Cmake. Install them via apt: `$ sudo apt install build-essential cmake`
2. You also need a Python environment. We tested it under conda Env and Python 3.9
3. One specific library HEAAN depends on is [GMP](https://ftp.gnu.org/gnu/gmp/). 
GMP 6.2.1 worked fine for us.

```bash
cd gmp-6.2.1
./configure SHARED=on 
make
make check  # GMP developers strongly recommend doing this
(sudo) make install 
```

If you encounter a *version number mismatch* error message like:

> GMP version check (6.2.1/6.2.0)  
> *** version number mismatch: inconsistency between gmp.h and libgmp

during building NTL library, try installing GMP library system-wide via apt instead.
```bash
sudo apt install libgmp3-dev 
sudo ldconfig
```

You also need the Number Theory Library [NTL](https://libntl.org/download.html). We tested NTL 11.5.1. NOTE that you need to add the flag `NTL_GMP_LIP=on` to Configure.

```bash
cd NTL/src
./configure SHARED=on NTL_GMP_LIP=on
make
make check  # optional
(sudo) make install
```

Now you are ready to build HEAAN. (need to be compiled with -fPIC)

```bash
cd HEAAN/lib
#    # Make sure in `src/subdir.mk`, there's the `-fPIC` option in g++ call (Line 59)
make 
# And run the test example
cd run
make
./TestHEAAN bootstrapping
```

Building SEAL
-------------

Provided that build-essential, cmake, and python are available, building SEAL library is simple.

SEAL requires more recent version of CMake than what is available on Ubuntu18.04.
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
