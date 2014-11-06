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
    test = np.array([uniform, ] * N).T
    sample = (test < right_end_points).astype('int')
    samples = np.zeros(num_samples)
    for i in xrange(num_samples):
        samples[i] = idx[np.searchsorted(sample[i], 1)]
    if num_samples == 1:
        return samples[0]
    return samples


def sample_from_bernoulli(p, M=1):
    """ returns N x M numpy array with M Bernoulli(p) N-bit samples """
    N = len(p)
    p = np.array(p)
    p /= p.sum()
    v_cp = np.array([p, ] * M).transpose()
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


def ltqnorm(p):
    """
    Modified from the author's original perl code (original comments follow below)
    by dfield@yahoo-inc.com.  May 3, 2004.

    Lower tail quantile for standard normal distribution function.

    This function returns an approximation of the inverse cumulative
    standard normal distribution function.  I.e., given P, it returns
    an approximation to the X satisfying P = Pr{Z <= X} where Z is a
    random variable from the standard normal distribution.

    The algorithm uses a minimax approximation by rational functions
    and the result has a relative error whose absolute value is less
    than 1.15e-9.

    Author:      Peter John Acklam
    Time-stamp:  2000-07-19 18:26:14
    E-mail:      pjacklam@online.no
    WWW URL:     http://home.online.no/~pjacklam
    """
    import math

    if p <= 0 or p >= 1:
        # The original perl code exits here, we'll throw an exception instead
        raise ValueError("Argument to ltqnorm %f must be in open interval (0,1)" % p)

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, \
         -2.759285104469687e+02, 1.383577518672690e+02, \
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, \
         -1.556989798598866e+02, 6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
         4.374664141464968e+00, 2.938163982698783e+00)
    d = ( 7.784695709041462e-03, 3.224671290700398e-01, \
          2.445134137142996e+00, 3.754408661907416e+00)

    # Define break-points.
    plow = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for upper region:
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

def find_latent_gaussian(mu, corr, accuracy=1e-10):
    """
    Compute parameters for the hidden Gaussian random vector U generating the
    binary Bernulli vector X with mean m and covariances c according to
    X = 0 <=> U < -g
    X = 1 <=> U > -g
    Adopted from www.kyb.mpg.de/bethgegroup/code/efficientsampling
    """

    from statsmodels.sandbox.distributions.multivariate import mvstdnormcdf

    if np.any(mu < 0) or np.any(mu >= 1):
        raise Exception("Mean for Gaussians has to be between 0 and 1!")

    n = len(mu)
    g = np.array([ltqnorm(m) for m in mu])
    L = np.eye(n)

    for i in xrange(n):
        for j in xrange(i + 1, n):
            c_min = -1
            c_max = 1

            # constant
            pn = mu[[i, j]].prod()

            # check whether DG distribution for covariance exists
            if (corr[i, j] - mvstdnormcdf(-g[[i, j]], np.array([np.inf, np.inf]), -1) + pn) < -1e-3 or \
               (corr[i, j] - mvstdnormcdf(-g[[i, j]], np.array([np.inf, np.inf]), 1) + pn) > 1e-3:
                raise Exception('A joint Bernoulli distribution with the given covariance matrix does not exist!')

            # determine Lambda_ij iteratively by bisection (Psi is monotonous in rho)
            while c_max - c_min > accuracy:
                c_new = (c_max + c_min) / 2.
                if corr[i, j] > mvstdnormcdf(-g[[i, j]], np.array([np.inf, np.inf]), c_new) - pn:
                    c_min = c_new
                else:
                    c_max = c_new

            L[i, j] = c_max
            L[j, i] = c_max

    return g.reshape(len(mu), 1), L


def sample_from_dichotomized_gaussian(mu, sigma, n, gamma=None, rho=None, accuracy=1e-10):

    from scipy.linalg import sqrtm

    return_inv = False
    if gamma is None:
        return_inv = True
        gamma, rho = find_latent_gaussian(mu, sigma, accuracy)

    sqrt_rho = np.real(sqrtm(rho))

    t = np.dot(sqrt_rho, np.random.randn(len(mu), n))
    x = t > np.repeat(-gamma, n, axis=1)
    x.dtype = np.uint8

    if return_inv:
        return x, gamma, rho
    else:
        return x
