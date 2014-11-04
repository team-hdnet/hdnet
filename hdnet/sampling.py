# -*- coding: utf-8 -*-
"""
    hdnet.sampling
    ~~~~~~~~~~~~~~

    Some simple routines for sampling from certain distributions.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np


def sample_from_prob_vector(p, num_samples=1):
    """ given numpy probability vector p on N states produce num_samples samples 
        returns: a (num_samples) integer vector with state labeled 0, ..., N-1
    """
    N = len(p)
    p = np.array(p)
    p /= p.sum()
    idx = p.argsort()
    sorted_p = p[idx]
    right_end_points = np.cumsum(sorted_p)
    uniform = np.random.random(num_samples)
    test = np.array([uniform,] * N).T
    sample = (test < right_end_points).astype('int')
    samples = np.zeros(num_samples)
    for i in xrange(num_samples):
        samples[i] = idx[np.searchsorted(sample[i], 1)]
    if num_samples == 1:
        return samples[0]
    return samples

def sample_from_Bernoulli(p, M=1):
    """ returns N x M numpy array with M Bernoulli(p) N-bit samples """
    N = len(p)
    p = np.array(p)
    p /= p.sum()
    v_cp = np.array([p,] * M).transpose()
    rand_vect = np.random.random((N, M))
    outcome = v_cp > rand_vect
    data = outcome.astype("int")
    if M == 1:
        return data[:, 0]
    return data

def energy(J, theta, x):
    """ Ising Energy of binary pattern x is:
                Ex = -.5 x^T[J-diag(J)]x + theta*x """
    return -.5 * np.dot(x, np.dot(J - np.diag(J.diagonal()), x)) + np.dot(theta, x)

def integer_to_binary(state, N):
    """ given state 0, ..., 2 **N - 1, returns corresponding binary vector x """
    return np.binary_repr(state, N)

def sample_from_ising(J, theta, num_samples=2):
    """ WARNING:  MIGHT NOT BE WORKING PROPERLY !!!
    
        given Ising model (J, theta) on N neurons produce num_samples samples 
        returns: a (N x num_samples) binary matrix with each column a binary vector (Ising sample)
    """
    N = len(theta)

    p = np.zeros(2 ** N)
    for i in xrange(2 ** N):
        x = np.array([np.int(k) for k in list(np.binary_repr(i, N))])
        p[i] = -energy(J, theta, x)
    p = np.exp(p)
    p /= p.sum()

    samples_int = sample_from_prob_vector(p, num_samples=num_samples)

    if num_samples == 1:
        return np.array([np.int(k) for k in list(np.binary_repr(samples_int, N))])

    samples = np.zeros((N, num_samples))
    for i in xrange(num_samples):
        samples[:, i] = np.array([np.int(k) for k in list(np.binary_repr(samples_int[i], N))])

    return samples
