

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy


# extensions = [
#     Extension("parallel", sources=["parallel.pyx"], include_dirs=[numpy.get_include()],
#      extra_compile_args=["-O3"], language="c++")
# ]

# setup(
#     name="processing_module",
#     ext_modules = cythonize(extensions),
# )
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("top_funcs",
              sources=["cython.pyx"],
                include_dirs=[numpy.get_include()],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              ) 
]

setup( 
  name = "bpr",
  cmdclass = {"build_ext": build_ext},
  ext_modules = cythonize(ext_modules)
)
