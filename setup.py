import os
from setuptools import setup#, find_packages
from distutils.sysconfig import get_python_inc
#import pathlib

# python include dir
py_include_dir = os.path.join(get_python_inc())

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
#from pybind11 import get_cmake_dir

cwd = os.getcwd()
__version__ = "1.01"

ext_modules1 = [
    Pybind11Extension("HEAAN",
        ["fase/bind/base64.cpp", "fase/bind/heaan_wrapper.cpp"],
        # Example: passing in the version to the compiled code
        include_dirs=[py_include_dir,
         'pybind11/include',
          '/usr/local/include',
           'fase/HEAAN/src'],
        language='c++',
        extra_compile_args=['-std=c++17'],
        extra_objects=['/usr/local/lib/libntl.so', 'fase/HEAAN/lib/libHEAAN.a'],
        define_macros = [('VERSION_INFO', __version__)],
        package_dir = {'': 'fase/'},
        )]

setup(
    name="fase",
    version=__version__,
    author="DeepInsight",
    #packages=find_packages(),
    author_email="hschoi@dinsight.ai",
    url="",
    description="FHE binder",
    long_description="",
    ext_modules=ext_modules1,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    force=True # force recompile the shared library. 
)

import subprocess
proc = subprocess.Popen("mv *.cpython*.so ./fase", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
