# -*- coding: utf-8 -*-
"""
    hdnet.hopfield
    ~~~~~~~~~~~~~~

    hopfield network python class with hopfield dynamics [Hopfield, PNAS, 1982]
    Built in learning/training of network is outer product learning rule (OPR)

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import numpy as np

# os.environ["C_INCLUDE_PATH"] = np.get_include()
# os.environ["NPY_NO_DEPRECATED_API"] = "!"
# try:
# import pyximport
#     pyximport.install()
# 
#     import dynamics
#     CYTHON = True
#     print "using cython"
# except:
#     CYTHON = False
#     print "not using cython"

def heaviside(X, dtype=None):
    """ given M x N numpy array, return Heaviside points-wise:
        H(r) = 1 if r > 0; else H(r) = 0
    """
    H = (np.sign(X).astype(int) + 1) // 2
    if dtype is None:
        return H
    else:
        return H.astype(dtype)


class HopfieldNet(object):
    """ Class for a binary Hopfield network

    Main Parameters:
    -----------
    N: int
        length of {0,1} binary vectors
    J: float array
        Hopfield connectivity matrix of network
    theta: array of thresholds for the nodes
    """

    def __init__(self, N=5, J=None, theta=None, name=None, update="asynchronous", symmetric=True):
        self.learn_iterations = 0  # how many learning steps have been taken so far
        self.N = N
        self.symmetric = symmetric
        if J is None:
            self.J = np.zeros((self.N, self.N))
        else:
            self.J = J
        if theta is None:
            self.theta = np.zeros(self.N)
        else:
            self.theta = theta
        self.name = name or self.__class__.__name__
        self.update = update
        self.neuron_order = xrange(self.N)
        self.last_num_iter_for_convergence = 0  # hopfield dynamics steps previous __call__ took

    # for saving network variables using self.savez() function
    _savevars = ["J", "N", "theta", "neuron_order", "learn_iterations", "update"]

    def reset(self):
        self.J = np.zeros((self.N, self.N))
        self.theta = np.zeros(self.N, dtype=float)
        self.last_num_iter_for_convergence = 0
        self.learn_iterations = 0  # how many learning steps have been taken so far

    def __call__(self, X, converge=True, max_iter=10 ** 5, clamped_nodes={}):
        """ Usage:  my_Hopfield_net(X) returns the Hopfield dynamics update to patterns
            stored in rows of M x N matrix X

            if converge = False then 1 update run through the neurons is performed
            else: Hopfield dynamics run on X until convergence OR max_iter iterations reached
            (NOTE: set max_iter = Inf to always force convergence)

            clamped_nodes is dictionary of those nodes not to update during the dynamics
        """
        ndim = X.ndim  # so that 1D vectors and arrays of vectors both work as X
        X = np.atleast_2d(X)

        out = np.zeros_like(X)
        niter = 0
        if converge:
            while not (X == out).all():
                if niter >= max_iter:
                    import warnings

                    warnings.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
                niter += 1
                out = X
                X = self.hopfield_binary_dynamics(
                    X, clamped_nodes=clamped_nodes, update=self.update)
            self.last_num_iter_for_convergence = niter
        else:
            self.last_num_iter_for_convergence = 1
            X = self.hopfield_binary_dynamics(X, clamped_nodes=clamped_nodes, update=self.update)

        if ndim == 1:
            return X.ravel()
        else:
            return X

    def learn_all(self, X):
        """ learning M patterns in Hopfield net using OPR [Hopfield, 82]

        X: (M, N)-dim array
            M binary patterns of length N to be stored
        """
        self.store_patterns_using_outer_products(X)

    def store_patterns_using_outer_products(self, X):
        self.reset()
        X = np.atleast_2d(X)
        S = 2 * X - 1
        for s in S:
            self.J += np.outer(s, s)
        self.J *= 2 / float(len(X))
        self.J[np.eye(self.N, dtype=bool)] *= 0

    def hopfield_binary_dynamics(self, X, update="asynchronous", clamped_nodes={}):  #, cython=CYTHON):
        """ applying Hopfield dynamics on X

            update can be "asynchronous" or "synchronous"

            clamped_nodes is dict of those nodes *not* to change in the dynamics
            (clamped does not work with cython right now)
        """
        # if cython:
        #     import dynamics
        #     
        #     dynamics.hopfield_binary_dynamics(X, X.shape[0], X.shape[1], self.J, self.theta)
        #     return X
        if update == "asynchronous":
            M, N = X.shape
            X_update = np.empty((M, N), dtype=X.dtype)
            X_update[:, :N] = X
            for bit in self.neuron_order:
                if bit not in clamped_nodes:
                    X_update[:, bit] = heaviside(
                        np.dot(self.J[bit, :], X_update.T) - self.theta[bit])
            ret = X_update[:, :self.N]
        elif update == "synchronous":
            ret = heaviside(np.dot(X, self.J.T) - self.theta[None, :])
        else:
            raise NotImplementedError

        return ret

    def savez(self, filename="network"):
        d = dict([(v, getattr(self, v)) for v in self._savevars])
        np.savez(filename, **d)

    def loadz(self, filename="network"):
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        d = np.load(base + ext)
        for v in self._savevars:
            setattr(self, v, d[v])

    def bits_recalled(self, X, converge=True):
        """ NEEDS TO BE SPED UP: CURRENTLY TOO SLOW"""
        return (X == self(X, converge=converge)).mean()

    def exact_recalled(self, X, converge=True):
        """ NEEDS TO BE SPED UP: CURRENTLY TOO SLOW"""
        return (X == self(X, converge=converge)).all(1).mean()

    def num_hopfield_iter(self, X, max_iter=10 ** 3):
        """ Returns array consisting of number of hopfield iterations to
        converge elements in X
        """
        count_arr = []
        for x in X:
            count = 1
            out = self.hopfield_binary_dynamics(x)
            while not (x == out).all():
                count += 1
                out = x
                x = self.hopfield_binary_dynamics(x)
                if count > max_iter:
                    import warnings

                    warnings.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
            count_arr.append(count)
        return count_arr

    def J_norm(self):
        """ vector of row 2-norms of J
        """
        return np.sqrt((self.J ** 2).sum(1))

    def compute_kappa(self, X):
        """ computes minimum marginal of dynamics update """
        S = 2 * X - 1
        Y = np.dot(X, self.W.T) - self.theta[None, :]
        return (S * Y / self.Wnorm).min()

    def energy(self, x):
        """ energy Ex = -.5 x^T[J-diag(J)]x + theta*x """
        return -.5 * np.dot(x, np.dot(self.J - np.diag(self.J.diagonal()), x)) + np.dot(self.theta, x)
