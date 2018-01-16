#!/usr/bin/python

from distutils.core import setup
from os.path import join
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
import os


if __name__ == '__main__':

    sources = ['ridge_sgd_fast.pyx',
               join('blas', 'ddot.c'),
               join('blas', 'daxpy.c'),
               join('blas', 'dscal.c')]
    includes = ['cblas', np.get_include()]

    cblas_libs = []
    if os.name == 'posix':
        cblas_libs.append('m')

    compile_args = ['-O3']
    ext_modules = cythonize([Extension('ridge_sgd_fast',
                                       sources=sources,
                                       include_dirs=includes,
                                       libraries=cblas_libs,
                                       extra_compile_args=compile_args,
                                       language='c++')])
    setup(ext_modules=ext_modules)
