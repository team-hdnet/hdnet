# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.hopfield
    ~~~~~~~~~~~~~~

    Classes providing Hopfield network functionality, both dynamics and training.
"""

from __future__ import print_function

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

    Parameters
    ----------
    N : int, optional
        Number of nodes in network (default None)
    J : numpy array, optional
        Coupling matrix of size N x N, where N denotes the number
        of nodes in the network (default None)
    theta : numpy array, optional
        Thresholds vector of size N, where N denotes the number
        of nodes in the network (default None)
    name : str, optional
        Name of network (default None)
    update : str, optional
        Type of Hopfield dynamics update, "synchronous"
        or "asynchronous" (default "asynchronous")
    symmetric : bool, optional
        Symmetric coupling matrix (default True)

    Returns
    -------
    network : :class:`.HopfieldNet`
        Instance of :class:`.HopfieldNet` class
    """

    _SAVE_ATTRIBUTES_V1 = ['_J', '_N', '_theta', '_neuron_order',
                           '_learn_iterations', '_update', '_symmetric']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'HopfieldNet'

    def __init__(self, N=None, J=None, theta=None, name=None, update="asynchronous", symmetric=True):
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
        self._neuron_order = range(self._N) if self._N else None
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
        for :meth:`num_nodes`.
        
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
        J : numpy array
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network (default None)
        """
        return self._J

    @property
    def J(self):
        """
        Returns the N x N matrix (with N denoting the number of nodes in the network)
        of coupling strengths of nodes in the network, shortcut for :meth:`coupling_matrix`.

        Returns
        -------
        J : numpy array
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network
        """
        return self._J

    @property
    def thresholds(self):
        """
        Returns a numpy vector of length N (with N denoting the number of nodes in the network)
        of thresholds for all nodes.

        Returns
        -------
        J : numpy array
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network
        """
        return self._theta

    @property
    def theta(self):
        """
        Returns a numpy vector of length N (with N denoting the number of nodes in the network)
        of thresholds for all nodes, shortcut for :meth:`thresholds`.

        Returns
        -------
        J : numpy array
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network
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
        Returns number of iterations needed in training
        phase until convergence of network parameters.

        Returns
        -------
        iterations : int
            Number of iterations until convergence
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
        Returns update flag as string, indicating Hopfield
        update type. Can be 'synchronous; or 'asynchronous'.
        
        Returns
        -------
        update : str
            Update flag
        """
        return self._update

    def reset(self):
        """
        Resets the network variables to base state
        (coupling strengths J, node thresholds theta
        and other status variables)

        .. note:

            If the network has been trained, this information will
            be lost!

        Returns
        -------
        Nothing
        """
        self._J = np.zeros((self._N, self._N))
        self._theta = np.zeros(self._N, dtype=float)
        self._last_num_iter_for_convergence = 0
        self._learn_iterations = 0  # how many learning steps have been taken so far

    def __call__(self, X, converge = True, max_iter = 10 ** 5, clamped_nodes = None, record_iterations = False,
                 record_energies = False):
        """
        Usage: network(X) returns the Hopfield dynamics update to patterns
        stored in rows of M x N matrix X. Calls :meth:`converge_dynamics`.
        """
        return self.converge_dynamics(X, converge = converge, max_iter = max_iter,
                                      clamped_nodes = clamped_nodes,
                                      record_iterations = record_iterations,
                                      record_energies = record_energies)

    def converge_dynamics(self, X, converge = True, max_iter = 10 ** 5, clamped_nodes = None, record_iterations = False,
                          record_energies = False):
        """
        Computes the Hopfield dynamics update to patterns stored in rows of M x N matrix.

        If `converge` is False then 1 update run through the neurons is performed,
        otherwise Hopfield dynamics are run on X until convergence or `max_iter`
        iterations of updates are reached.

        .. note:
            Set max_iter = Inf to always force convergence

        `clamped_nodes` is dictionary of those nodes not to update during the dynamics.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        converge : bool, optional
            Flag whether to converge Hopfield dynamics. If False,
            just one step of dynamics is performed (default True)
        max_iter : int, optional
            Maximal number of iterations of dynamics (default 10 ** 5)
        clamped_nodes : list, optional
            List of clamped nodes that are left untouched during
            dynamics update (default None)
        record_iterations : bool, optional
            If `True`, function records number of Hopfield dynamics
            update steps needed for converge of input to Hopfield memory
            for each input data vector and returns it as second return
            argument (default False)
        record_energies : bool, optional
            If `True`, function records difference in Ising energy for each
            update step needed for convergence of input to Hopfield memory
            for each input data vector and returns it as third return
            argument (default False)

        Returns
        -------
        patterns : numpy array
            Converged patterns (memories) of Hopfield dynamics of input
            argument X
        iters : numpy array
            Number of dynamics iterations needed to converge to memory
        energies : numpy array
            Ising energy reduction upon convergence to memory
        """
        if clamped_nodes is None:
            clamped_nodes = {}

        ndim = X.ndim  # so that 1D vectors and arrays of vectors both work as X
        X = np.atleast_2d(X)

        out = np.zeros_like(X)
        niter = 0
        if record_iterations:
            niters = np.zeros((X.shape[0],), dtype = np.int)
        if record_energies:
            energies = np.zeros((X.shape[0],), dtype = np.double)
            old_energies = self.energy(X)
        if converge:
            while (niter == 0) or not (X == out).all():
                if niter >= max_iter:
                    hdlog.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
                niter += 1
                out = X
                Xnew = self.hopfield_binary_dynamics(
                    X, clamped_nodes=clamped_nodes, update=self._update)
                if record_iterations:
                    niters += (Xnew != X).astype(np.int).max(axis = 1)
                if record_energies:
                    new_energies = self.energy(Xnew)
                    energies += (old_energies - new_energies)
                    old_energies = new_energies
                X = Xnew
            self._last_num_iter_for_convergence = niter
        else:
            self._last_num_iter_for_convergence = 1
            Xnew = self.hopfield_binary_dynamics(X, clamped_nodes=clamped_nodes, update=self._update)
            if record_iterations:
                niters += (Xnew != X).astype(np.int).max(axis = 1)
            if record_energies:
                new_energies = self.energy(Xnew)
                energies += (old_energies - new_energies)
            X = Xnew

        if ndim == 1:
            X = X.ravel()

        if record_iterations and record_energies:
            return X, niters, energies
        elif record_iterations:
            return X, niters
        elif record_energies:
            return X, energies
        else:
            return X

    def learn_all(self, X, disp=False):
        """
        Learning M patterns in Hopfield network using outer product learning
        rule (OPR) [Hopfield, 82]

        Interface method, calls :meth:`store_patterns_using_outer_products`.

        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N to be stored,
            where N is the number of nodes in the network
        disp : bool, optional
            Display training log messages (default False)

        Returns
        -------
        Nothing
        """
        self.store_patterns_using_outer_products(X)

    def store_patterns_using_outer_products(self, X):
        """
        Store patterns in X using outer product learning rule (OPR).
        Sets coupling matrix J.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N to be stored,
            where N is the number of nodes in the network
        
        Returns
        -------
        Nothing
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
        Applying Hopfield dynamics on X.
        Update can be "asynchronous" (default) or "synchronous".
        clamped_nodes is dict of those nodes *not* to change in the dynamics

        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        update : str, optional
            Type of Hopfield dynamics update (default "asynchronous")
        clamped_nodes : dict, optional
            Dictionary of nodes to leave untouched during dynamics
            update (default None)
        
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
        Returns fraction of correctly recalled bits on input data `X`.

        Parameters
        ----------
        X : 2d numpy array, int
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        converge : bool, optional
            Flag whether to converge Hopfield dynamics. If False,
            just one step of dynamics is performed (default True)

        Returns
        -------
        recalled : float
            Fraction of correctly recalled bits
        """
        return float((X == self(X, converge=converge)).mean())

    def exact_recalled(self, X, converge=True):
        """
        Returns fraction of raw patterns stored as memories in
        unmodified form.
        
        Parameters
        ----------
        X : 2d numpy array, int
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        converge : bool, optional
            Flag whether to converge Hopfield dynamics. If False,
            just one step of dynamics is performed (default True)
        
        Returns
        -------
        fraction : float
            Fraction of exactly stored memories
        """
        return (X == self(X, converge=converge)).all(1).mean()

    def num_hopfield_iter(self, X, max_iter=10 ** 5):
        """
        Returns array consisting of the number of Hopfield iterations
        needed to converge elements in `X` to their memories.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        max_iter : int, optional
            Maximal number if iterations to perform per element (default 10 ** 5)
        
        Returns
        -------
        count : numpy array
            Number of iterations performed for each element in `X`
        """
        count_arr = []
        for x in X:
            count = 1
            out = self(x)
            while not (x == out).all():
                count += 1
                out = x
                x = self(x)
                if count > max_iter:
                    hdlog.warn("Exceeded maximum number of iterations (%d)" % max_iter)
                    break
            count_arr.append(count)
        return count_arr

    def J_norm(self):
        """
        Returns vector of row 2-norms of coupling matrix J
        of Hopfield network.
        
        Returns
        -------
        norm : 1d numpy array, float
            Vector of 2-norms of coupling matrix
        """
        return np.sqrt((self._J ** 2).sum(1))

    def compute_kappa(self, X):
        """
        Computes minimum marginal of dynamics update.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network

        Returns
        -------
        Value : Type
            Description
        """
        S = 2 * X - 1
        Y = np.dot(X, self.W.T) - self._theta[None, :]
        return (S * Y / self.Wnorm).min()

    def energy(self, x):
        r"""
        Calculates the energy of a pattern ``x`` according to the
        Hopfield network.

        The energy of a pattern ``x`` computes as:

        .. math:: E(x) = -\frac{1}{2} x^T \cdot [J - \text{diag}(J)] \cdot x + \theta\cdot x
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        
        Returns
        -------
        energy : float
            Energy of input pattern according to Hopfield network.
        """
        X = np.atleast_2d(x)
        energies = []
        for x in X:
            energies.append(-.5 * np.dot(x, np.dot(self._J - np.diag(self._J.diagonal()), x)) + np.dot(self._theta, x))
        if len(energies) == 1:
            return np.array(energies[0], dtype = np.double)
        else:
            return np.array(energies, dtype = np.double)

    # representation

    def __repr__(self):
        return '<HopfieldNetwork: {n} nodes>'.format(n=self._N)

    # i/o

    def save(self, file_name='hopfield_network', extra=None):
        """
        Saves Hopfield network to file.
        
        Parameters
        ----------
        file_name : str, optional
            File name to save network to (default 'hopfield_network')
        extra : dict, optional
            Extra information to save to file (default None)
        
        Returns
        -------
        Nothing
        """
        super(HopfieldNet, self)._save(file_name=file_name,
                                      attributes=self._SAVE_ATTRIBUTES_V1,
                                      version=self._SAVE_VERSION,
                                      extra=extra)

    @classmethod
    def load(cls, file_name='hopfield_network', load_extra=False):
        """
        Loads Hopfield network from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            net = HopfieldNet.load('file_name')
        
        Parameters
        ----------
        file_name : str, optional
            File name to load network from (default 'hopfield_network')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)
        
        Returns
        -------
        network : :class:`.HopfieldNet`
            Instance of :class:`.HopfieldNet` if loaded, `None` upon error
        """
        return super(HopfieldNet, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        # internal function to load v1 file format
        hdlog.debug('loading HopfieldNet, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)


class HopfieldNetMPF(HopfieldNet):
    r"""
    Hopfield network, with training using Minimum Energy Flow (MEF)
    Hillar, Christopher and Sohl-Dickstein, Jascha and Koepsell, Kilian, 
    Efficient and optimal binary Hopfield associative memory storage using minimum probability flow, 2012. 
    https://arxiv.org/abs/1204.2916v1
    """

    def learn_all(self, X, disp=False):
        """
        Learn from M memory samples with Minimum Energy Flow (MEF)

        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)

        Returns
        -------
        Nothing
        """
        self.store_patterns_using_mpf(np.asarray(X), disp=disp)

    def objective_function(self, X, J=None):
        """
        Note: accepts J with -2 theta on the diagonal
        Returns the MPF objective function evaluated over patterns X
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        J : numpy array, optional
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network (default None)

        Returns
        -------
        objective_func : numpy array
            MEF objective function evaluated over patterns X
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
        J : numpy array, optional
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network (default None)
        
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
        for batch in range(nbatch):
            X = sampler(batch_size)
            S = 2 * X - 1
            Kfull = np.exp(-S * np.dot(X, J.T) + .5 * np.diag(J)[None, :])
            Ksum += Kfull.sum()
        return Ksum / (nbatch * batch_size)

    def objective_gradient(self, X, J=None, return_K=False):
        """
        Computes MEF objective gradient on input data X given coupling
        strengths J.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        J : numpy array, optional
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network (default None)
        return_K : bool, optional
            Flag wether to return K (default False)
        
        Returns
        -------
        dJ [, K] : numpy array [, numpy array]
            Update to coupling matrix J [and K if return_K is True]
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
        J : numpy array, optional
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network (default None)
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
        for batch in range(nbatch):
            hdlog.debug("batch %i/%i" % (batch + 1,nbatch))
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
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        
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
        J : numpy array
            Coupling matrix of size N x N, where N denotes the number
            of nodes in the network
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

    def store_patterns_using_mpf(self, X, disp=False, **kwargs):
        """
        Stores patterns in X using Minimum Energy Flow (MEF) learning
        rule.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)

        Returns
        -------
        status : dict
            Dictionary containing status information
        """
        # TODO: document
        # TODO: status printing?
        import scipy.optimize
        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(
            self.objective_gradient_minfunc, self._J.ravel(), args=[X],
            iprint=-1 if not disp else 0, **kwargs)
        # A,Amin,status = scipy.optimize.fmin_l_bfgs_b(
        # self.objective_gradient_minfunc, np.zeros(self.N * self.N,), args=[X])

        J = A.reshape(self._N, self._N)
        self._theta = -.5 * np.diag(J)
        self._J = J
        self._J[np.eye(self._N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * len(X)
        self._learn_iterations = status["learn_iterations"]
        return status

    # parameters: 
    # X: patterns to be learned
    # r: radius

    def store_patterns_using_r_mpf(self, X, r = 1, p = .1, m = 10, disp=False, **kwargs):
        """
        Stores patterns in X using generalized Minimum Energy Flow (MEF) learning
        rule, r - MEF.
        
        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        r : int, optional
            Radius of Hamming ball (default 1)
        m : int, optional
            Flipping probability of bits of data pattern (default 10)
        p : float, optional
            Flipping probability of bits of data pattern (default .1)
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)

        Returns
        -------
        status : dict
            Dictionary containing status information
        """
        import math
        import scipy.optimize

        # TODO check whether this works as expected for r = 1

        # TODO rework parameters r, p, m -- and investigate on good standard values

        def objective_gradient_minfunc(J, X, r = r):
            # TODO: vectorize as much as possible
            J = J.reshape(self._N, self._N)
            Xt = 2 * X - 1
            S = np.array([np.diag(2 * (np.random.random(self.N) < p).astype(int) - 1) for _ in range(m)])
            T = np.zeros((m, self.N, self.N))
            for k in range(m):
                T[k] = J - np.dot(np.dot(S[k].T, J), S[k])
            s = 0
            for i, x in enumerate(X):
                # this is SLOOOW, at least cythonize after testing it does the right thing
                e1 = math.exp(self.energy(x))
                dot1 = np.tensordot(x.T, T, axes = 1)
                dot2 = np.tensordot(dot1, x, axes = 1)
                dotexp = np.exp(dot2)
                s += np.sum(dotexp)
            #TODO: return gradient of J here, dJ
            #return s, dJ (or dJ, s), check docs for scipy.optimize.fmin_l_bfgs_b
            return s

        # TODO set approx_grad = False once gradient computation in place
        A, Amin, status = scipy.optimize.fmin_l_bfgs_b(
            objective_gradient_minfunc, self._J.ravel(), args=[X],
            iprint=-1 if not disp else 0, approx_grad = True, **kwargs)

        J = A.reshape(self._N, self._N)
        self._theta = -.5 * np.diag(J)
        self._J = J
        self._J[np.eye(self._N, dtype=bool)] *= 0
        status["learn_iterations"] = status["funcalls"] * len(X)
        self._learn_iterations = status["learn_iterations"]
        return status

    def learn_from_sampler(self, sampler, sample_size, batch_size=None, use_gpu=False):
        """
        Learn from sampler
        
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
        return '<HopfieldNetwork: {n} nodes, MEF training>'.format(n=self._N)


# end of source
