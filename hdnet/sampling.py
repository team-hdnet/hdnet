# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.sampling
    ~~~~~~~~~~~~~~

    Some simple routines for sampling from certain distributions.

"""

from __future__ import print_function

import numpy as np


def sample_from_prob_vector(p, num_samples=1):
    """
    Given numpy probability vector p on N states produce num_samples samples
    returns: a (num_samples) integer vector with state labeled 0, ..., N-1
    
    Parameters
    ----------
    p : Type
        Description
    num_samples : int, optional
        Description (default 1)
    
    Returns
    -------
    Value : Type
        Description
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
    for i in range(num_samples):
        samples[i] = idx[np.searchsorted(sample[i], 1)]
    if num_samples == 1:
        return samples[0]
    return samples


def sample_from_bernoulli(p, M=1):
    """
    Returns N x M numpy array with M Bernoulli(p) N-bit samples
    
    Parameters
    ----------
    p : Type
        Description
    M : int, optional
        Description (default 1)
    
    Returns
    -------
    Value : Type
        Description
    """
    N = len(p)
    p = np.array(p)
    v_cp = np.array([p, ] * M).transpose()
    rand_vect = np.random.random((N, M))
    outcome = v_cp > rand_vect
    data = outcome.astype("int")
    if M == 1:
        return data[:, 0]
    return data


def energy(J, theta, x):
    """
    Ising Energy of binary pattern x is:
        Ex = -.5 x^T[J-diag(J)]x + theta*x
    
    Parameters
    ----------
    J : Type
        Description
    theta : Type
        Description
    x : Type
        Description
    
    Returns
    -------
    Value : Type
        Description
    """
    return -.5 * np.dot(x, np.dot(J - np.diag(J.diagonal()), x)) + np.dot(theta, x)


def integer_to_binary(state, N):
    """
    Given state 0, ..., 2\*\*N - 1, returns corresponding binary vector x
    
    Parameters
    ----------
    state : Type
        Description
    N : Type
        Description
    
    Returns
    -------
    Value : Type
        Description
    """
    return np.binary_repr(state, N)

def sample_from_ising_exact(J, theta, num_samples):
    """
    Given an Ising model `J`, `theta` on N neurons produces `num_samples` samples
    Returns: a (N x num_samples) binary matrix with each column a binary vector (Ising sample)

    .. warning:

    MIGHT NOT BE WORKING PROPERLY!

    Parameters
    ----------
    J : Type
    Description
    theta : Type
    Description
    num_samples : int, optional
    Description (default 2)

    Returns
    -------
    Value : Type
    Description
    """
    N = len(theta)

    p = np.zeros(2 ** N)
    for i in range(2 ** N):
        x = np.array([np.int(k) for k in list(np.binary_repr(i, N))])
        p[i] = -energy(J, theta, x)
        p = np.exp(p)
        p /= p.sum()

    samples_int = sample_from_prob_vector(p, num_samples=num_samples)

    if num_samples == 1:
        return np.array([np.int(k) for k in list(np.binary_repr(samples_int, N))])

    samples = np.zeros((N, num_samples))
    for i in range(num_samples):
        samples[:, i] = np.array([np.int(k) for k in list(np.binary_repr(samples_int[i], N))])

    return samples

def sample_from_ising_metropolis(J, theta, num_samples, burn_in = None, k = None):
    """
    Given an Ising model `J`, `theta` on N sites produces `num_samples` samples
    from the model using a (MCMC) Metropolis sampler.

    Parameters
    ----------
    J : 2d numpy array
        Coupling strengths of Ising model (symmetric, 0 diagonal values)
    theta : 1d numpy array
        Site biases of Ising model
    num_samples : int
        Number of samples to draw
    burn_in : int, optional
        Burn in time of Markov chain (default 100*N)
    k: float, optional
        Scales probability mass for neurons
    Returns
    -------
    X : 2d numpy array
        array of dimensions N x num_samples containing N samples
    """

    N = len(theta)

    if burn_in is None:
        burn_in = 100 * N
    
    if k is None:
        k = 0.01
    
    n_sampling_steps = burn_in + num_samples 

    # samples
    X = []

    # starting vector
    xold = np.asarray([int(i) for i in np.binary_repr(np.random.randint(0,np.power(2,N)),N)])
    Eold = -np.inner(theta,xold)+np.inner(xold,np.dot(J,xold))
    pm = np.exp(-k*np.arange(1,N)); pm /= np.sum(pm)
    for t in range(n_sampling_steps):
        m = np.random.choice(N-1,size=1,p=pm)+1
        foo = np.random.choice(N,size=m,replace=False)
        xnew = np.zeros(N)
        for i in range(N):
            if i in foo:
                xnew[i] = 1-xold[i]
            else:
                xnew[i] = xold[i]
        Enew = np.inner(-theta,xnew)+np.inner(xnew,np.dot(J,xnew))
        dE = Enew-Eold
        acceptance_ratio = np.exp(-dE)
        u = np.random.uniform()
        if u<acceptance_ratio:
            Eold = Enew
            if t>burn_in-1:
                X.append(xnew)
            xold = xnew
        else:
            if t>burn_in-1:
                X.append(xold)
    return np.transpose(np.asarray(X))

def sample_from_ising_gibbs(J, theta, num_samples, burn_in = None, sampling_steps = None):
    """
    Given an Ising model `J`, `theta` on N sites produces `num_samples` samples
    from the model using a (MCMC) Gibbs sampler.
    Inspired by a Matlab implementation of a Gibbs sampler by J. Sohl-Dickstein.

    Parameters
    ----------
    J : 2d numpy array
        Coupling strengths of Ising model (symmetric, 0 diagonal values)
    theta : 1d numpy array
        Site biases of Ising model
    num_samples : int
        Number of samples to draw
    burn_in : int, optional
        Burn in time of Markov chain (default 100*N)
    sampling_steps : int, optional
        Number of Markov steps in between samples (default 10*N)

    Returns
    -------
    X : 2d numpy array
        array of dimensions N x num_samples containing N samples
    """
    N = len(theta)

    if burn_in is None:
        burn_in = 100 * N

    if sampling_steps is None:
        sampling_steps = 10 * N
    
    n_sampling_steps = burn_in + (num_samples - 1) * sampling_steps

    # sampling dimensions
    dimensions = np.random.random_integers(0, N - 1, n_sampling_steps)

    # random numbers
    rand = np.random.random((n_sampling_steps, 1))

    # samples
    X = np.zeros((N, num_samples))

    # starting vector
    x = np.random.random((N, 1)) * 2

    idx = 0
    next_sample = burn_in

    for si in range(n_sampling_steps):
        E_active = 2 * np.dot(J[dimensions[si], :], x) + theta[dimensions[si]]
        p_active = 1. / (1. + np.exp(E_active)) # NB. sigmoid s: s(-E_active)
        if p_active > rand[si]:
            x[dimensions[si]] = 1
        else:
            x[dimensions[si]] = 0
        
        if si == next_sample:
            next_sample += sampling_steps
            X[:, idx] = x.ravel()
            idx += 1

    return X


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
    WWW URL:     `<http://home.online.no/~pjacklam>`_
    
    Returns
    -------
    Value : Type
        Description
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


def ltqnorm_nd(arr):
    """
    Missing documentation
    
    Returns
    -------
    Value : Type
        Description
    """
    if len(arr) == 0:
        return arr
    shape = arr.shape
    return np.array(map(ltqnorm, arr.ravel())).reshape(shape)


def find_latent_gaussian(bin_means, bin_cov, accuracy=1e-10):
    """
    Compute parameters for the hidden Gaussian random vector U generating the
    binary Bernulli vector X with mean m and covariances c according to
    X = 0 <=> U < -g
    X = 1 <=> U > -g

    Adapted from `<www.kyb.mpg.de/bethgegroup/code/efficientsampling>`_
    
    Parameters
    ----------
    bin_means : Type
        Description
    bin_cov : Type
        Description
    accuracy : int, optional
        Description (default 1e-10)
    
    Returns
    -------
    Value : Type
        Description
    """
    from statsmodels.sandbox.distributions.multivariate import mvstdnormcdf

    if np.any(bin_means < 0) or np.any(bin_means >= 1):
        raise Exception("Mean for Gaussians has to be between 0 and 1!")

    d = len(bin_means)
    gauss_mean = np.array([ltqnorm(m) for m in bin_means])
    gauss_cov = np.eye(d)

    for i in range(d):
        for j in range(i + 1, d):
            c_min = -1
            c_max = 1

            # constant
            pn = bin_means[[i, j]].prod()

            # check whether DG distribution for covariance exists
            if (bin_cov[i, j] - mvstdnormcdf(-gauss_mean[[i, j]], np.array([np.inf, np.inf]), -1) + pn) < -1e-3 or \
               (bin_cov[i, j] - mvstdnormcdf(-gauss_mean[[i, j]], np.array([np.inf, np.inf]), 1) + pn) > 1e-3:
                raise Exception('A joint Bernoulli distribution with the given covariance matrix does not exist!')

            # determine Lambda_ij iteratively by bisection (Psi is monotonous in rho)
            while c_max - c_min > accuracy:
                c_new = (c_max + c_min) / 2.
                if bin_cov[i, j] > mvstdnormcdf(-gauss_mean[[i, j]], np.array([np.inf, np.inf]), c_new) - pn:
                    c_min = c_new
                else:
                    c_max = c_new

            gauss_cov[i, j] = gauss_cov[j, i] = c_max

    return gauss_mean.reshape(len(bin_means), 1), gauss_cov


def sample_from_dichotomized_gaussian(bin_means, bin_cov, num_samples, gauss_means=None, gauss_cov=None, accuracy=1e-10):
    """
    Missing documentation
    
    Parameters
    ----------
    bin_means : Type
        Description
    bin_cov : Type
        Description
    num_samples : Type
        Description
    gauss_means : Type, optional
        Description (default None)
    gauss_cov : Type, optional
        Description (default None)
    accuracy : int, optional
        Description (default 1e-10)
    
    Returns
    -------
    Value : Type
        Description
    """
    from scipy.linalg import sqrtm

    return_inv = False
    if gauss_means is None:
        return_inv = True
        gauss_means, gauss_cov = find_latent_gaussian(bin_means, bin_cov, accuracy)

    sqrt_cov = np.real(sqrtm(gauss_cov))
    t = np.dot(sqrt_cov, np.random.randn(len(bin_means), num_samples))
    samples = t > np.repeat(-gauss_means, num_samples, axis=1)
    samples.dtype = np.uint8

    if return_inv:
        return samples, gauss_means, gauss_cov
    else:
        return samples


def poisson_marginals(means, accuracy=1e-10):
    """
    Finds the probability mass functions (pmfs) and approximate supports of a set of
    Poisson random variables with means specified in input "means". The
    second argument, "acc", specifies the desired degree of accuracy. The
    "support" is taken to consist of all values for which the pmfs is greater
    than acc.

    Inputs:
    means: the means of the Poisson RVs
    acc: desired accuracy

    Outputs:
    pmfs: a cell-array of vectors, where the k-th element is the probability
    mass function of the k-th Poisson random variable.
    supports: a cell-array of vectors, where the k-th element is a vector of
    integers of the states that the k-th Poisson random variable would take
    with probability larger than "acc". E.g., P(kth
    RV==supports{k}(1))=pmfs{k}(1);

    Code from the paper: 'Generating spike-trains with specified
    correlations', Macke et al., submitted to Neural Computation

    Adapted from `<http://www.kyb.mpg.de/bethgegroup/code/efficientsampling>`_
    
    Parameters
    ----------
    means : Type
        Description
    accuracy : int, optional
        Description (default 1e-10)
    
    Returns
    -------
    Value : Type
        Description
    """
    from scipy.stats import poisson
    import math

    cmfs = []
    pmfs = []
    supps = []

    for k in range(len(means)):
        cmfs.append(poisson.cdf(range(0, int(max(math.ceil(5 * means[k]), 20) + 1)), means[k]))
        pmfs.append(poisson.pmf(range(0, int(max(math.ceil(5 * means[k]), 20) + 1)), means[k]))
        supps.append(np.where((cmfs[k] <= 1 - accuracy) & (pmfs[k] >= accuracy))[0])
        cmfs[k] = cmfs[k][supps[k]]
        pmfs[k] = poisson.pmf(supps[k], means[k])

    return np.array(pmfs), np.array(cmfs), np.array(supps)


def dg_second_moment(u, gauss_mean1, gauss_mean2, support1, support2):
    """
    Missing documentation
    
    Parameters
    ----------
    u : Type
        Description
    gauss_mean1 : Type
        Description
    gauss_mean2 : Type
        Description
    support1 : Type
        Description
    support2 : Type
        Description
    
    Returns
    -------
    Value : Type
        Description
    """
    # subfunction DGSecondMoment: Calculate second Moment of the DG
    # for a given correlation lambda
    #a very, very inefficient function for calculating the second moments of a
    #DG with specified gammas and supports and correlation lambda
    from statsmodels.sandbox.distributions.multivariate import mvnormcdf

    sig = np.array([[1, u], [u, 1]])
    x, y = np.meshgrid(support2, support1)
    xy = x * y

    ps = np.zeros((len(support1), len(support2)))

    for i in range(len(support1)):
        for j in range(len(support2)):
            ps[i, j] = mvnormcdf([gauss_mean1[i], gauss_mean2[j]], [0, 0], sig)

    ps2 = ps.copy()
    for i in range(len(support1)):
        for j in range(len(support2)):
            if i > 0 and j > 0:
                ps2[i, j] = ps[i, j] + ps[i - 1, j - 1] - ps[i - 1, j] - ps[i, j - 1]
            elif j > 0 and i == 0:
                ps2[i, j] = ps[i, j] - ps[i, j - 1]
            elif i > 0 and j == 0:
                ps2[i, j] = ps[i, j] - ps[i - 1, j]
            elif i == 0 and j == 0:
                ps2[i, j] = ps[i, j]

    ps2 = np.clip(ps2, 0, ps2.max())
    joint = ps2.copy()
    ps2 = ps2 * xy
    secmom = np.sum(ps2)
    return secmom, joint


def find_dg_any_marginal(pmfs, bin_cov, supports, accuracy=1e-10):
    """
    [gammas,Lambda,joints2D] = FindDGAnyMarginal(pmfs,Sigma,supports)
    Finds the paramters of a Multivariate Discretized Gaussian with specified marginal
    distributions and covariance matrix

    Inputs:
    pmfs: the probability mass functions of the marginal distribution of the
    input-random variables. Must be a cell-array with n elements, each of
    which is a vector which sums to one
    Sigma: The covariance matrix of the input-random variable. The function
    does not check for admissability, i.e. results might be wrong if there
    exists no random variable which has the specified marginals and
    covariance.
    supports: The support of each dimension of the input random variable.
    Must be a cell-array with n elements, each of whcih is a vector with
    increasing entries giving the possible values of each random variable,
    e.g. if the first dimension of the rv is 1 with probability .2, 3 with
    prob .8, then pmfs{1}=[.2,.8], supports{1}=[1,3]; If no support is
    specified, then each is taken to be [0:numel(pdfs{k}-1];

    Outputs:
    gammas: the discretization thresholds, as described in the paper. When
    sampling. The k-th dimension of the output random variable is f if e.g.
    supports{k}(1)=f and gammas{k}(f) <= U(k) <= gammas{k}(f+1)
    Lambda: the covariance matrix of the latent Gaussian random variable U
    joints2D: An n by n cell array, where each entry contains the 2
    dimensional joint distribution of  a pair of dimensions of the DG.

    Code from the paper: 'Generating spike-trains with specified
    correlations', Macke et al., submitted to Neural Computation

    Adapted from `<http://www.kyb.mpg.de/bethgegroup/code/efficientsampling>`_
    
    Parameters
    ----------
    pmfs : Type
        Description
    bin_cov : Type
        Description
    supports : Type
        Description
    accuracy : int, optional
        Description (default 1e-10)
    
    Returns
    -------
    Value : Type
        Description
    """
    from scipy.optimize import minimize_scalar

    #keyboard
    d = len(pmfs)

    if supports is None:
        supports = []

    cmfs = []
    mu = []
    gammas = []

    for i in range(d):
        # take default supports if only one argument is specified
        if len(supports) < i:
            supports.append(range(len(pmfs[k]) for k in range(len(pmfs))))

        supports[i] = supports[i].ravel()
        pmfs[i] = pmfs[i].ravel()
        cmfs.append(np.cumsum(pmfs[i]))

        mu.append(np.dot(supports[i], pmfs[i]))

        gammas.append(ltqnorm_nd(cmfs[i]))
        if bin_cov[i, i] <= 0 or np.isnan(bin_cov[i, i]):
            bin_cov[i, i] = np.dot(supports[i] ** 2, pmfs[i]) - mu[i] ** 2

    
    #numerics for finding the off-diagonal entries. Very inefficient
    #and use of the optimization toolbox is really an overkill for finding the
    #zero-crossing of a one-dimensional funcition on a compact interval
    
    lam = np.zeros((d, d)) * np.nan
    joints = {}
    
    for i in range(d):
        lam[i, i] = 1
        joints[i, i] = pmfs[i]
        for j in range(i + 1, d):
            #fprintf('Finding Lambda(#d #d)\n',i,j)
            moment = bin_cov[i, j] + mu[i] * mu[j]
            #take the correlation coefficient between the two dimensions as
            #starting point
            # x0 = bin_cov[i, j] / math.sqrt(bin_cov[i, i]) / math.sqrt(bin_cov[j, j])

            #minimized squared difference between the specified covariance speccov and the
            #covariance of the DG
            #not optimized for speed yet, in fact, it is terrible for speed. For
            #example, one really easy thing would be to evalulate the cholesky of the
            #covariance (in the bivariate gaussian cdf) only once, and not multiple
            #times
            minidiff = lambda x: \
                (dg_second_moment(x if x > -1 and x < 1 else -1 + .0000000001 if x <= -1 else 1 - .0000000001,
                gammas[i], gammas[j], supports[i], supports[j])[0] - moment) ** 2

            mx = minimize_scalar(minidiff, method='Bounded', bounds=(-1, 1), options = {'xatol': accuracy})['x']

            lam[i, j] = lam[j, i] = mx
            _, joint = dg_second_moment(mx, gammas[i], gammas[j], supports[i], supports[j])

            joints[i, j] = joint
            joints[j, i] = joint.T

    return gammas, lam, joints


def sample_dg_any_marginal(gauss_means, gauss_cov, num_samples, supports=None):
    """
    [samples,hists]=SampleDGAnyMarginal(gammas,Lambda,supports,Nsamples)
    Generate samples for a Multivariate Discretized Gaussian with parameters
    gammas" and "Lambda" and "supports". The number of samples generated is "Nsamples"

    input and output arguments are as described in "DGAnyMarginal"

    Usage:
    Code from the paper: 'Generating spike-trains with specified
    correlations', Macke et al., submitted to Neural Computation

    Adapted from `<http://www.kyb.mpg.de/bethgegroup/code/efficientsampling>`_
    
    Parameters
    ----------
    gauss_means : Type
        Description
    gauss_cov : Type
        Description
    num_samples : Type
        Description
    supports : Type, optional
        Description (default None)
    
    Returns
    -------
    Value : Type
        Description
    """
    d = gauss_cov.shape[0]

    if supports is None:
        supports = []
        for i in range(d):
            supports.append(range(len(gauss_means[i])))

    cc = np.linalg.cholesky(gauss_cov).T
    B = np.dot(np.random.randn(num_samples, d), cc)

    hists = []
    samples = np.zeros((num_samples, d))
    for i in range(d):
        bins = np.hstack((-np.inf, gauss_means[i], np.inf))
        h = np.histogram(B[:, i], bins=bins)[0]
        hists.append(h / float(num_samples))
        bin_idx = np.digitize(B[:, i], bins) - 1
        samples[:, i] = supports[i][bin_idx]
        hists[i] = hists[i][0:max(0, len(hists[i]) - 1)]

    return samples, np.array(hists)


# end of source
