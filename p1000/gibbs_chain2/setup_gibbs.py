# Builds gibbs_p1000_core.pyx in place (same -O3 flag as the %%cython cell in the notebook).
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension("gibbs_p1000_core", ["gibbs_p1000_core.pyx"], extra_compile_args=["-O3"])
setup(name="gibbs_p1000_core",
      ext_modules=cythonize([ext], language_level=3),
      script_args=["build_ext", "--inplace"])
