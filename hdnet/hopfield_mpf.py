# -*- coding: utf-8 -*-
"""
    hdnet.hopfield_mpf
    ~~~~~~~~~~~~~~~~~~

    Hopfield network using Minimum Probability Flow (MPF) (Sohl-Dickstein, Battaglino, Deweese, 2009)
    for training / learning of binary patterns

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np

from hopfield import HopfieldNet


class HopfieldNetMPF(HopfieldNet):
    """ Class for a Hopfield network with MPF learning rule

    Parameters:
    -----------
    N: int
        length of {0,1} binary vectors
    J: float array
        Hopfield connectivity matrix of network
    theta: array of thresholds for the nodes

    Conventions:
        Note: J_{ij} for i not j = weight W_{ij} between neuron i and j
              J_{ii} = 0
        where (Hopfield) dynamics on input x are
            Out_i(x) = H(\sum_{j not i} J_{ij}x_j - theta_i)
        Here, H = Heaviside function:  H(r) = 1 r > 0, H(r) = 0 if r <= 0.

        Note energy function is:
            Energy(x) = -.5 x^T[J-diag(J)]x + theta*x

    """

    def reset(self):
        super(HopfieldNetMPF, self).reset()

    def learn_all(self, X):
        """ learning of M memory samples with MPF

        X: (M, N)-dim array
            M  N-bit patterns to be stored
        """
        self.store_patterns_using_mpf(np.asarray(X))

    def objective_function(self, X, J=None):
        """ Note: accepts J with -2 theta on the diagonal
            Returns the MPF objective function evaluated over patterns X
        """
        if J is None:
            J = self.J.copy()
            J[np.eye(self.N, dtype=bool)] = -2 * self.theta
        S = 2 * X - 1
        Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
        # J2 = J * J
        return Kfull.sum() / len(np.atleast_2d(X))

    def objective_function_batched(self, sampler, sample_size, batch_size, randstate, J=None):
        """ This is to be able to fit network with more samples X than can be held in memory at once
        """
        np.random.set_state(randstate)
        nbatch = sample_size / batch_size
        if J is None:
            J = self.J
            J[np.eye(self.N, dtype=bool)] = -2 * self.theta
        Ksum = 0
        for batch in xrange(nbatch):
            X = sampler(batch_size)
            S = 2 * X - 1
            Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
            Ksum += Kfull.sum()
        return Ksum / (nbatch * batch_size)

    def objective_gradient(self, X, J=None, return_K=False):
        """ J is a square np.array
            X is a M x N np.array of binary vectors """
        if J is None:
            J = self.J
            J[np.eye(self.N, dtype=bool)] = -2 * self.theta
        X = np.atleast_2d(X)
        M, N = X.shape
        S = 2 * X - 1
        Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
        dJ = -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
        if self.symmetric is True:
            dJ = .5 * (dJ + dJ.T)
        if return_K:
            return Kfull.sum() / M, dJ / M
        else:
            return dJ / M

    def objective_gradient_batched(self, sampler, sample_size, batch_size, randstate,
                                   J=None, return_K=False):
        np.random.set_state(randstate)
        nbatch = sample_size / batch_size
        if J is None:
            J = self.J.copy()
            J[np.eye(self.N, dtype=bool)] = -2 * self.theta
        Ksum = 0
        dJ = np.zeros((self.N, self.N), dtype=float)
        for batch in xrange(nbatch):
            # print "batch %i/%i" % (batch+1,nbatch)
            X = sampler(batch_size)
            S = 2 * X - 1
            Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
            dJ += -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
            Ksum += Kfull.sum()
        if self.symmetric is True:
            dJ = .5 * (dJ + dJ.T)
        M = nbatch * batch_size
        if return_K:
            return Ksum / M, dJ / M
        else:
            return dJ / M

    def objective_gradient_minfunc(self, J, X):
        K, dJ = self.objective_gradient(X, J=J.reshape(self.N, self.N), return_K=True)
        return K, dJ.ravel()

    def objective_gradient_minfunc_batched(self, J, sampler, sample_size, batch_size, randstate):
        K, dJ = self.objective_gradient_batched(J=J.reshape(self.N, self.N), return_K=True,
                                                sampler=sampler, sample_size=sample_size, batch_size=batch_size,
                                                randstate=randstate)
        return K, dJ.ravel()

    def optcallback(p):
        print "."

    def store_patterns_using_mpf(self, X):
        import scipy.optimize

        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(
            self.objective_gradient_minfunc, self.J.ravel(), args=[X])
        # A,Amin,status = scipy.optimize.fmin_l_bfgs_b(
        # self.objective_gradient_minfunc, np.zeros(self.N * self.N,), args=[X])

        J = A.reshape(self.N, self.N)
        self.theta = -.5 * np.diag(J)
        self.J = J
        self.J[np.eye(self.N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * len(X)
        self.learn_iterations = status["learn_iterations"]
        return status

    def learn_from_sampler(self, sampler, sample_size, batch_size=None, use_gpu=False):
        """ To learn from a sampler """
        if use_gpu:
            raise NotImplementedError
        if batch_size is None:
            batch_size = sample_size
        import scipy.optimize

        randstate = np.random.get_state()
        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(self.objective_gradient_minfunc_batched,
                                                       self.J.ravel(),
                                                       args=[sampler, sample_size, batch_size, randstate])
        J = A.reshape(self.N, self.N)
        self.theta = -.5 * np.diag(J)
        self.J = J
        self.J[np.eye(self.N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * batch_size * (sample_size / batch_size)
        self.learn_iterations = status["learn_iterations"]
        return status


# end of source
