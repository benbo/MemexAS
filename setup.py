from distutils.core import setup
from Cython.Build import cythonize
#python2.7 setup.py build_ext --inplace
setup(
  name = 'Fast n-grams',
  ext_modules = cythonize("MyNgrams.pyx"),
)

