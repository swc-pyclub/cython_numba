# -*- coding: utf-8 -*-
"""

compile extension via "python setup.py build_ext -i"
or use pyximport (see below)

Created on Mon Jan 15 21:02:47 2018

@author: arne
"""


import numpy as np
import timeit
import matplotlib.pyplot as plt
from collections import OrderedDict
from numba.decorators import autojit

from ridge_sgd_fast import (
    ridge_sgd_cython,
    ridge_sgd_cython_types,
    ridge_sgd_cython_memoryviews,
    ridge_sgd_cython_pointers,
    ridge_sgd_cython_vectorized,
    ridge_sgd_cython_vectorized_types,
    ridge_sgd_cython_blas
)

# uncomment for automatic compilation on import
# import pyximport
# pyximport.install()


def ridge_sgd_naive(X, y, w, alpha, perm):

    D = X.shape[1]
    for t, i in enumerate(perm):

        gamma = 1. / (1 + alpha*t)

        # regularization step
        for j in range(D):
            w[j] *= (1. - gamma * alpha)

        # loss step
        z = 0
        for j in range(D):
            z += w[j] * X[i, j]

        for j in range(D):
            w[j] += gamma * X[i, j] * (z - y[i])


def ridge_sgd_vectorized(X, y, w, alpha, perm):

    for t, i in enumerate(perm):

        i = perm[t]
        gamma = 1. / (1 + alpha*t)

        # regularization step
        w *= (1. - gamma * alpha)

        # loss step
        z = np.dot(w, X[i, :])
        w += gamma * X[i, :] * (z - y[i])


ridge_sgd_numba = autojit(ridge_sgd_naive)


def create_data(N=10000, D=1000):

    X = np.random.randn(N, D)
    w_true = np.sin(2*np.pi*np.linspace(0, 1, D))
    y = np.dot(X, w_true)

    return X, y, w_true


def run_benchmark():

    X, y, w_true = create_data()

    setup = """
import numpy as np
from __main__ import (
    create_data, ridge_sgd_naive,
    ridge_sgd_vectorized,
    ridge_sgd_cython,
    ridge_sgd_cython_types,
    ridge_sgd_cython_memoryviews,
    ridge_sgd_cython_pointers,
    ridge_sgd_numba,
    ridge_sgd_cython_vectorized,
    ridge_sgd_cython_vectorized_types,
    ridge_sgd_cython_blas
)
X, y, w_true = create_data(%d, 500)
w = np.zeros_like(w_true)
alpha = 1.
n_epochs = 1.
T = int(X.shape[0] * n_epochs)
perm = np.random.randint(0, X.shape[0], T)
"""

    functions = OrderedDict()
    functions.update({'ridge_sgd_naive': 'Python\n(naive)'})
    functions.update({'ridge_sgd_cython': 'Cython'})
    functions.update({'ridge_sgd_vectorized': 'Python\n(vector.)'})
    functions.update({'ridge_sgd_cython_vectorized': 'Cython\n(vector.)'})
    functions.update({'ridge_sgd_cython_types': 'Cython\n(types)'})
    functions.update({'ridge_sgd_cython_memoryviews': 'Cython\n(memviews)'})
    functions.update({'ridge_sgd_cython_pointers': 'Cython\n(pointers)'})
    functions.update({'ridge_sgd_cython_blas': 'Cython\n(BLAS)'})
    functions.update({'ridge_sgd_numba': 'Numba'})

    sample_sizes = [500, 1000, 2000, 4000, 8000, 16000]
#    sample_sizes = [20, 40, 80, 160, 320, 640]  # for testing

    n_fun = len(functions)
    n_repeat = 5
    n_sizes = len(sample_sizes)
    runtime = np.zeros((n_fun, n_sizes, n_repeat))
    for i, fun in enumerate(functions):
        for j, size in enumerate(sample_sizes):
            t = timeit.repeat("%s(X, y, w, alpha, perm)" % fun, setup % size,
                              repeat=n_repeat, number=10)
            runtime[i, j, :] = t
            print "%s (%d): %0.5f s" % (fun, size, np.mean(t))

    # Plot summary
    plt.rc('axes', color_cycle=['b', 'g', 'r', 'c', 'm', 3*[.25], 'y',
                                [1, .5, 0], 3*[.6]])
    fun_names = [functions[name] for name in functions.keys()]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, xscale='log', yscale='log')
    ax2 = fig.add_subplot(1, 2, 2, xscale='log', yscale='log')

    xx = sample_sizes
    yy_ref = np.mean(runtime[0, :, :], axis=1)
    for i, fun_name in enumerate(fun_names):

        yy = np.mean(runtime[i, :, :], axis=1)
        yerr = np.std(runtime[i, :, :], axis=1)

        ax1.errorbar(xx, yy, yerr=yerr, fmt='o-', linewidth=2, label=fun_name,
                     alpha=.8)
        ax2.plot(xx, yy_ref / yy, 'o-', linewidth=2, label=fun_name,
                 alpha=.8)

    ax1.set_xlabel('Sample size')
    ax1.set_ylabel('Run time (s)')
    ax1.set_xlim(100, 1.2*np.max(sample_sizes))
    ax1.grid()
    leg = ax1.legend(loc=2, numpoints=1, fontsize=9)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0)

    ax2.set_xlabel('Sample size')
    ax2.set_ylabel('Speedup')
    ax2.set_xlim(100, 1.2*np.max(sample_sizes))
    ax2.grid()
    leg = ax2.legend(loc=2, numpoints=1, fontsize=9)
    leg.get_frame().set_alpha(0)
    leg.get_frame().set_linewidth(0)

    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=10)

    fig.tight_layout()

    fig_file = 'ridge_sgd_benchmark.'
    for ff in ['pdf', 'svg']:
        fig.savefig(fig_file + ff, format=ff)

    plt.show()


if __name__ == '__main__':
    run_benchmark()

