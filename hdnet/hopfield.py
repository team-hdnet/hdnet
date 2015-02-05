# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.hopfield
    ~~~~~~~~~~~~~~

    Classes providing Hopfield network functionality, both dynamics and training.
"""

import numpy as np
from hdnet.maths import heaviside
from hdnet.util import Restoreable, hdlog

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


class HopfieldNet(Restoreable, object):
    """
    Hopfield network class with binary hopfield dynamics [Hopfield, PNAS, 1982]
    Built in learning/training of network is outer product learning rule (OPR).

    Main Parameters:
    N: int: length of {0,1} binary vectors
    J: float array: Hopfield connectivity matrix of network
    theta: array of thresholds for the nodes
    """

    _SAVE_ATTRIBUTES_V1 = ['_J', '_N', '_theta', '_neuron_order',
                           '_learn_iterations', '_update', '_symmetric']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'HopfieldNet'

    def __init__(self, N=None, J=None, theta=None, name=None, update="asynchronous", symmetric=True):
        """
        Missing documentation
        
        Parameters
        ----------
        N : Type, optional
            Description (default None)
        J : Type, optional
            Description (default None)
        theta : Type, optional
            Description (default None)
        name : Type, optional
            Description (default None)
        update : str, optional
            Description (default "asynchronous")
        symmetric : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
        object.__init__(self)
        Restoreable.__init__(self)
        self._learn_iterations = 0  # how many learning steps have been taken so far
        self._N = N
        self._symmetric = symmetric
        if J is None and N > 0:
            self._J = np.zeros((self._N, self._N))
        else:
            self._J = J
        if theta is None and N > 0:
            self._theta = np.zeros(self._N)
        else:
            self._theta = theta
        self._name = name or self.__class__.__name__
        self._update = update
        self._neuron_order = xrange(self._N) if self._N else None
        self._last_num_iter_for_convergence = 0  # hopfield dynamics steps previous __call__ took
        self._learn_iterations = 0  # how many learning steps have been taken so far


    @property
    def num_nodes(self):
        """
        Returns the number of nodes in the network.

        Returns
        -------
        n : int
        """
        return self._N

    @property
    def N(self):
        """
        Returns the number of nodes in the network, shortcut
        for :meth:`~HopfieldNet.num_nodes`.
        
        Returns
        -------
        n : int
        """
        return self._N

    @property
    def coupling_matrix(self):
        """
        Returns the N x N matrix (with N denoting the number of nodes in the network)
        of coupling strengths of nodes in the network.

        Returns
        -------
        J : 2d numpy array
        """
        return self._J

    @property
    def J(self):
        """
        Returns the N x N matrix (with N denoting the number of nodes in the network)
        of coupling strengths of nodes in the network, shortcut for :meth:`~HopfieldNet.coupling_matrix`.

        Returns
        -------
        J : 2d numpy array
        """
        return self._J

    @property
    def thresholds(self):
        """
        Returns a numpy vector of length N (with N denoting the number of nodes in the network)
        of thresholds for all nodes.

        Returns
        -------
        J : 2d numpy array
        """
        return self._theta

    @property
    def theta(self):
        """
        Returns a numpy vector of length N (with N denoting the number of nodes in the network)
        of thresholds for all nodes, shortcut for :meth:`~HopfieldNet.thresholds`.

        Returns
        -------
        J : 2d numpy array
        """
        return self._theta

    @property
    def neuron_order(self):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        return self._neuron_order

    @property
    def learn_iterations(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._learn_iterations

    @property
    def symmetric(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._symmetric

    @property
    def update(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._update

    def reset(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        self._J = np.zeros((self._N, self._N))
        self._theta = np.zeros(self._N, dtype=float)
        self._last_num_iter_for_convergence = 0
        self._learn_iterations = 0  # how many learning steps have been taken so far

    def __call__(self, X, converge=True, max_iter=10 ** 5, clamped_nodes=None):
        """
        Usage:  my_Hopfield_net(X) returns the Hopfield dynamics update to patterns
        stored in rows of M x N matrix X

        if converge = False then 1 update run through the neurons is performed
        else: Hopfield dynamics run on X until convergence OR max_iter iterations reached
        (NOTE: set max_iter = Inf to always force convergence)

        clamped_nodes is dictionary of those nodes not to update during the dynamics
        
        Parameters
        ----------
        X : Type
            Description
        converge : bool, optional
            Description (default True)
        max_iter : int, optional
            Description (default 10 ** 5)
        clamped_nodes : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        if clamped_nodes is None:
            clamped_nodes = {}

        ndim = X.ndim  # so that 1D vectors and arrays of vectors both work as X
        X = np.atleast_2d(X)

        out = np.zeros_like(X)
        niter = 0
        if converge:
            while not (X == out).all():
                if niter >= max_iter:
                    hdlog.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
                niter += 1
                out = X
                X = self.hopfield_binary_dynamics(
                    X, clamped_nodes=clamped_nodes, update=self._update)
            self._last_num_iter_for_convergence = niter
        else:
            self._last_num_iter_for_convergence = 1
            X = self.hopfield_binary_dynamics(X, clamped_nodes=clamped_nodes, update=self._update)

        if ndim == 1:
            return X.ravel()
        else:
            return X

    def learn_all(self, X):
        """
        learning M patterns in Hopfield net using OPR [Hopfield, 82]

        X: (M, N)-dim array
            M binary patterns of length N to be stored
        
        Parameters
        ----------
        X : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        self.store_patterns_using_outer_products(X)

    def store_patterns_using_outer_products(self, X):
        """
        Missing documentation
        
        Parameters
        ----------
        X : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        self.reset()
        X = np.atleast_2d(X)
        S = 2 * X - 1
        for s in S:
            self._J += np.outer(s, s)
        self._J *= 2 / float(len(X))
        self._J[np.eye(self._N, dtype=bool)] *= 0

    def hopfield_binary_dynamics(self, X, update="asynchronous", clamped_nodes=None):
        """
        applying Hopfield dynamics on X

        update can be "asynchronous" or "synchronous"

        clamped_nodes is dict of those nodes *not* to change in the dynamics
        (clamped does not work with cython right now)
        
        Parameters
        ----------
        X : Type
            Description
        update : str, optional
            Description (default "asynchronous")
        clamped_nodes : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        # if cython:
        #     import dynamics
        #
        #     dynamics.hopfield_binary_dynamics(X, X.shape[0], X.shape[1], self.J, self.theta)
        #     return X
        if clamped_nodes is None:
            clamped_nodes = {}
        if update == "asynchronous":
            M, N = X.shape
            X_update = np.empty((M, N), dtype=X.dtype)
            X_update[:, :N] = X
            for bit in self._neuron_order:
                if bit not in clamped_nodes:
                    X_update[:, bit] = heaviside(
                        np.dot(self._J[bit, :], X_update.T) - self._theta[bit])
            ret = X_update[:, :self._N]
        elif update == "synchronous":
            ret = heaviside(np.dot(X, self._J.T) - self._theta[None, :])
        else:
            raise NotImplementedError

        return ret

    def bits_recalled(self, X, converge=True):
        """
        NEEDS TO BE SPED UP: CURRENTLY TOO SLOW
        
        Parameters
        ----------
        X : Type
            Description
        converge : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
        return (X == self(X, converge=converge)).mean()

    def exact_recalled(self, X, converge=True):
        """
        NEEDS TO BE SPED UP: CURRENTLY TOO SLOW
        
        Parameters
        ----------
        X : Type
            Description
        converge : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
        return (X == self(X, converge=converge)).all(1).mean()

    def num_hopfield_iter(self, X, max_iter=10 ** 3):
        """
        Returns array consisting of number of hopfield iterations to
        converge elements in X
        
        Parameters
        ----------
        X : Type
            Description
        max_iter : int, optional
            Description (default 10 ** 3)
        
        Returns
        -------
        Value : Type
            Description
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
                    hdlog.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
            count_arr.append(count)
        return count_arr

    def J_norm(self):
        """
        vector of row 2-norms of J
        
        Returns
        -------
        Value : Type
            Description
        """
        return np.sqrt((self._J ** 2).sum(1))

    def compute_kappa(self, X):
        """
        computes minimum marginal of dynamics update
        
        Parameters
        ----------
        X : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        S = 2 * X - 1
        Y = np.dot(X, self.W.T) - self._theta[None, :]
        return (S * Y / self.Wnorm).min()

    def energy(self, x):
        """
        energy Ex = -.5 x^T[J-diag(J)]x + theta*x
        
        Parameters
        ----------
        x : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        return -.5 * np.dot(x, np.dot(self._J - np.diag(self._J.diagonal()), x)) + np.dot(self._theta, x)

    # representation

    def __repr__(self):
        return '<HopfieldNetwork: {n} nodes>'.format(n=self._N)

    # i/o

    def save(self, file_name='hopfield_network', extra=None):
        """
        Missing documentation
        
        Parameters
        ----------
        file_name : str, optional
            Description (default 'hopfield_network')
        extra : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        super(HopfieldNet, self)._save(file_name=file_name,
                                      attributes=self._SAVE_ATTRIBUTES_V1,
                                      version=self._SAVE_VERSION,
                                      extra=extra)

    @classmethod
    def load(cls, file_name='hopfield_network', load_extra=False):
        """
        Missing documentation
        
        Parameters
        ----------
        file_name : str, optional
            Description (default 'hopfield_network')
        load_extra : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        return super(HopfieldNet, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('loading HopfieldNet, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)


class HopfieldNetMPF(HopfieldNet):
    """
    Hopfield network, with training using Minimum Probability Flow (MPF)
    (Sohl-Dickstein, Battaglino, Deweese, 2009) for training / learning of binary patterns

    Parameters:
    N: int
    length of {0,1} binary vectors
    J: float array
    Hopfield connectivity matrix of network
    theta: array of thresholds for the nodes

    Conventions:
    Note: J_{ij} for i not j = weight W_{ij} between neuron i and j, J_{ii} = 0
    where (Hopfield) dynamics on input x are Out_i(x) = H(\sum_{j not i} J_{ij}x_j - theta_i).
    Here, H = Heaviside function:  H(r) = 1 r > 0, H(r) = 0 if r <= 0.

    Note energy function is: Energy(x) = -.5 x^T[J-diag(J)]x + theta*x
    """

    def reset(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        super(HopfieldNetMPF, self).reset()

    def learn_all(self, X):
        """
        learning of M memory samples with MPF

        X: (M, N)-dim array
            M  N-bit patterns to be stored
        
        Parameters
        ----------
        X : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        self.store_patterns_using_mpf(np.asarray(X))

    def objective_function(self, X, J=None):
        """
        Note: accepts J with -2 theta on the diagonal
            Returns the MPF objective function evaluated over patterns X
        
        Parameters
        ----------
        X : Type
            Description
        J : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        if J is None:
            J = self._J.copy()
            J[np.eye(self._N, dtype=bool)] = -2 * self._theta
        S = 2 * X - 1
        Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
        # J2 = J * J
        return Kfull.sum() / len(np.atleast_2d(X))

    def objective_function_batched(self, sampler, sample_size, batch_size, randstate, J=None):
        """
        This is to be able to fit network with more samples X than can be held in memory at once
        
        Parameters
        ----------
        sampler : Type
            Description
        sample_size : Type
            Description
        batch_size : Type
            Description
        randstate : Type
            Description
        J : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        np.random.set_state(randstate)
        nbatch = sample_size / batch_size
        if J is None:
            J = self._J
            J[np.eye(self._N, dtype=bool)] = -2 * self._theta
        Ksum = 0
        for batch in xrange(nbatch):
            X = sampler(batch_size)
            S = 2 * X - 1
            Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
            Ksum += Kfull.sum()
        return Ksum / (nbatch * batch_size)

    def objective_gradient(self, X, J=None, return_K=False):
        """
        J is a square np.array
        X is a M x N np.array of binary vectors
        
        Parameters
        ----------
        X : Type
            Description
        J : Type, optional
            Description (default None)
        return_K : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        if J is None:
            J = self._J
            J[np.eye(self._N, dtype=bool)] = -2 * self._theta
        X = np.atleast_2d(X)
        M, N = X.shape
        S = 2 * X - 1
        Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
        dJ = -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
        if self._symmetric is True:
            dJ = .5 * (dJ + dJ.T)
        if return_K:
            return Kfull.sum() / M, dJ / M
        else:
            return dJ / M

    def objective_gradient_batched(self, sampler, sample_size, batch_size, randstate,
                                   J=None, return_K=False):
        """
        Missing documentation
        
        Parameters
        ----------
        sampler : Type
            Description
        sample_size : Type
            Description
        batch_size : Type
            Description
        randstate : Type
            Description
        J : Type, optional
            Description (default None)
        return_K : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        np.random.set_state(randstate)
        nbatch = sample_size / batch_size
        if J is None:
            J = self._J.copy()
            J[np.eye(self._N, dtype=bool)] = -2 * self._theta
        Ksum = 0
        dJ = np.zeros((self._N, self._N), dtype=float)
        for batch in xrange(nbatch):
            hdlog.debug("batch %i/%i" % (batch+1,nbatch))
            X = sampler(batch_size)
            S = 2 * X - 1
            Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
            dJ += -np.dot(X.T, Kfull * S) + .5 * np.diag(Kfull.sum(0))
            Ksum += Kfull.sum()
        if self._symmetric is True:
            dJ = .5 * (dJ + dJ.T)
        M = nbatch * batch_size
        if return_K:
            return Ksum / M, dJ / M
        else:
            return dJ / M

    def objective_gradient_minfunc(self, J, X):
        """
        Missing documentation
        
        Parameters
        ----------
        J : Type
            Description
        X : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        K, dJ = self.objective_gradient(X, J=J.reshape(self._N, self._N), return_K=True)
        return K, dJ.ravel()

    def objective_gradient_minfunc_batched(self, J, sampler, sample_size, batch_size, randstate):
        """
        Missing documentation
        
        Parameters
        ----------
        J : Type
            Description
        sampler : Type
            Description
        sample_size : Type
            Description
        batch_size : Type
            Description
        randstate : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        K, dJ = self.objective_gradient_batched(J=J.reshape(self._N, self._N), return_K=True,
                                                sampler=sampler, sample_size=sample_size, batch_size=batch_size,
                                                randstate=randstate)
        return K, dJ.ravel()

    def optcallback(p):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        pass

    def store_patterns_using_mpf(self, X, **kwargs):
        """
        Missing documentation
        
        Parameters
        ----------
        X : Type
            Description
        kwargs : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        # TODO: status printing?
        import scipy.optimize
        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(
            self.objective_gradient_minfunc, self._J.ravel(), args=[X], **kwargs)
        # A,Amin,status = scipy.optimize.fmin_l_bfgs_b(
        # self.objective_gradient_minfunc, np.zeros(self.N * self.N,), args=[X])

        J = A.reshape(self._N, self._N)
        self._theta = -.5 * np.diag(J)
        self._J = J
        self._J[np.eye(self._N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * len(X)
        self._learn_iterations = status["learn_iterations"]
        return status

    def learn_from_sampler(self, sampler, sample_size, batch_size=None, use_gpu=False):
        """
        Missing documentation
        
        Parameters
        ----------
        sampler : Type
            Description
        sample_size : Type
            Description
        batch_size : Type, optional
            Description (default None)
        use_gpu : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        """ To learn from a sampler """
        if use_gpu:
            raise NotImplementedError
        if batch_size is None:
            batch_size = sample_size
        import scipy.optimize

        randstate = np.random.get_state()
        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(self.objective_gradient_minfunc_batched,
                                                       self._J.ravel(),
                                                       args=[sampler, sample_size, batch_size, randstate])
        J = A.reshape(self._N, self._N)
        self._theta = -.5 * np.diag(J)
        self._J = J
        self._J[np.eye(self._N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * batch_size * (sample_size / batch_size)
        self._learn_iterations = status["learn_iterations"]
        return status

    # representation

    def __repr__(self):
        return '<HopfieldNetwork: {n} nodes, MPF training>'.format(n=self._N)


# end of source
