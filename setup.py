from setuptools import setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os
import subprocess
from pathlib import Path


experiment_directory = "{}/.experiments".format(Path(__file__).parent)
if not os.path.exists(experiment_directory):
  subprocess.call(['mkdir', experiment_directory])

ext_modules=[
    Extension("topology.cython.top_funcs",
              sources=["topology/cython/cython.pyx"],
                include_dirs=[numpy.get_include()],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]



setup(name='topology',
      version='0.0',
      packages=['topology'],
      cmdclass = {"build_ext": build_ext},
      ext_modules = cythonize(ext_modules)
      )
