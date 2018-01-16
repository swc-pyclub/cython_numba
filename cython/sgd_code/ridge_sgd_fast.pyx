#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INT

cdef extern double ddot_(int *N, double *X, int *INCX, double *Y, int *INCY)
cdef extern int dscal_(int *n, double *sa, double *sx, int *incx)
cdef extern int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
                       int *incy)

np.import_array()


def ridge_sgd_cython(X, y, w, alpha, perm):

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


def ridge_sgd_cython_vectorized(X, y, w, alpha, perm):

    for t, i in enumerate(perm):

        i = perm[t]
        gamma = 1. / (1 + alpha*t)

        # regularization step
        w *= (1. - gamma * alpha)

        # loss step
        z = np.dot(w, X[i, :])
        w += gamma * X[i, :] * (z - y[i])


def ridge_sgd_cython_types(np.ndarray[DOUBLE, ndim=2] X,
                           np.ndarray[DOUBLE, ndim=1] y,
                           np.ndarray[DOUBLE, ndim=1] w, double alpha,
                           np.ndarray[INT, ndim=1] perm):

    cdef int D = X.shape[1]
    cdef int i, j, t
    cdef double gamma, z

    for t, i in enumerate(perm):

        gamma = 1. / (1. + alpha*t)

        # regularization step
        for j in range(D):
            w[j] *= (1. - gamma * alpha)

        # loss step
        z = 0
        for j in range(D):
            z += w[j] * X[i, j]

        for j in range(D):
            w[j] += gamma * X[i, j] * (z - y[i])


def ridge_sgd_cython_memoryviews(double[:, ::1] X,
                                 double[:] y,
                                 double[:] w,
                                 double alpha,
                                 long[:] perm):

    cdef int D = X.shape[1]
    cdef int i, j, t
    cdef double gamma, z

    for t, i in enumerate(perm):

        i = perm[t]
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


def ridge_sgd_cython_vectorized_types(np.ndarray[DOUBLE, ndim=2] X,
                                      np.ndarray[DOUBLE, ndim=1] y,
                                      np.ndarray[DOUBLE, ndim=1] w,
                                      double alpha,
                                      np.ndarray[INT, ndim=1] perm):

    cdef int D = X.shape[1]
    cdef int i, t
    cdef double gamma, z

    for t, i in enumerate(perm):

        i = perm[t]
        gamma = 1. / (1 + alpha*t)

        # regularization step
        w *= (1. - gamma * alpha)

        # loss step
        z = np.dot(w, X[i, :])
        w += gamma * X[i, :] * (z - y[i])


def ridge_sgd_cython_pointers(np.ndarray[DOUBLE, ndim=2] X,
                              np.ndarray[DOUBLE, ndim=1] y,
                              np.ndarray[DOUBLE, ndim=1] w, double alpha,
                              np.ndarray[INT, ndim=1] perm):

    cdef int D = X.shape[1]
    cdef long i, t, j
    cdef double learn_rate,  z

    cdef double *Xp = <double*> X.data
    cdef double *yp = <double*> y.data
    cdef double *wp = <double*> w.data
    cdef long *pp = <long*> perm.data

    for t, i in enumerate(perm):

        gamma = 1. / (1. + alpha*t)

        # regularization step
        for j in range(D):
            wp[j] *= (1. - gamma * alpha)

        # loss step
        z = 0
        for j in range(D):
            z += wp[j] * Xp[i*D + j]

        for j in range(D):
            wp[j] += gamma * Xp[i*D + j] * (z - yp[i])


def ridge_sgd_cython_blas(np.ndarray[DOUBLE, ndim=2] X,
                          np.ndarray[DOUBLE, ndim=1] y,
                          np.ndarray[DOUBLE, ndim=1] w, double alpha,
                          np.ndarray[INT, ndim=1] perm):

    cdef int D = X.shape[1]
    cdef long i, t, j
    cdef double learn_rate, z, u

    cdef double *Xp = <double*> X.data
    cdef double *yp = <double*> y.data
    cdef double *wp = <double*> w.data
    cdef long *pp = <long*> perm.data

    cdef int incrx = 1
    cdef int incry = 1

    for t, i in enumerate(perm):

        gamma = 1. / (1. + alpha*t)

        # regularization step
        z = 1. - gamma*alpha
        dscal_(&D, &z, wp, &incrx)

        # loss step
        z = ddot_(&D, wp, &incrx, &Xp[i*D], &incry)
        u = gamma * (z - yp[i])
        daxpy_(&D, &u, wp, &incrx, &Xp[i*D], &incry)

