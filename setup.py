from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("fast_distance_computation.pyx")
)