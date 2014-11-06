# -*- coding: utf-8 -*-
"""
    hdnet.spikes_model
    ~~~~~~~~~~~~~~~~~~

    Null-models for spikes' statistics.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np
from time import time as now

from spikes import Spikes
from patterns import Patterns
from counter import Counter
from learner import Learner
from sampling import sample_from_bernoulli, sample_from_ising, sample_from_dichotomized_gaussian, find_latent_gaussian, \
    poisson_marginals


class SpikeModel(object):
    """ generic model of spikes (and stimulus)

    Parameters
        spikes: spikes to model
        stimulus: corresp stimulus if exists
        window_size: length of time window in binary bins
    """

    def __init__(self, spikes=None, stimulus=None, window_size=1, learner=None):
        self.spikes = spikes
        self.stimulus = stimulus
        self.window_size = window_size
        self.learner = learner or None

        if spikes is not None:
            self.original_spikes = spikes

    def fit(self, trials=None, remove_zeros=True, reshape=False):
        # TODO: take care of remove_zeros
        self.sample_spikes = self.sample_from_model(trials=trials, reshape=reshape)
        self.learner = Learner(spikes=self.sample_spikes)
        self.learner.learn_from_spikes(remove_zeros=remove_zeros)

    def chomp(self):
        # print "Chomping samples from model"
        self.emperical = Counter()
        self.emperical.chomp_spikes(spikes=self.sample_spikes)
        print "%d-bit (emperical): %d patterns (H = %1.3f)" % (
            self.sample_spikes.N, len(self.emperical), self.emperical.entropy())

        # print "Chomping dynamics (from network learned on the samples) applied to samples"
        self.memories = Patterns(learner=self.learner)
        self.memories.chomp_spikes(spikes=self.sample_spikes)
        print "%d-bit (hopnet): %d memories (H = %1.3f)" % (
            self.sample_spikes.N, len(self.memories), self.memories.entropy())

        # print "Before dynamics:"
        # print self.sample_spikes.spikes_arr
        # print "Applied dynamics:"
        hop_model_spikes = self.memories.apply_dynamics(spikes=self.sample_spikes, reshape=True)
        ma_err = np.abs(self.sample_spikes.spikes_arr - hop_model_spikes).mean()
        #        print hop_model_spikes
        print "Mean prediction: %1.4f/1.0 (vs guess zero: %1.4f)" % (
            (1 - ma_err), 1 - np.abs(self.sample_spikes.spikes_arr).mean())
        # # distortion
        # self.sample_spikes

    def distinct_patterns_over_windows(self, window_sizes=[1], trials=None, save_couplings=False, remove_zeros=True):
        """ Returns tuple: counts, entropies [, couplings]
                counts, entropies: arrays of size 2 x T x WSizes 
            (0: emperical from model sample, 1: dynamics from learned model on sample)"""
        trials = trials or range(self.original_spikes.T)
        counts = np.zeros((2, len(trials), len(window_sizes)))
        entropies = np.zeros((2, len(trials), len(window_sizes)))

        couplings = {}

        tot_learn_time = 0

        for ws, window_size in enumerate(window_sizes):
            couplings[window_size] = []

            for c, trial in enumerate(trials):
                print "Trial %d | ws %d" % (trial, window_size)

                self.window_size = window_size

                t = now()
                self.fit(trials=[trial], remove_zeros=remove_zeros)
                diff = now() - t
                print "[%1.3f min]" % (diff / 60.)
                tot_learn_time += diff

                if save_couplings:
                    couplings[ws].append(self.learner.network.J.copy())

                self.chomp()
                entropies[0, c, ws] = self.emperical.entropy()
                counts[0, c, ws] = len(self.emperical)
                entropies[1, c, ws] = self.memories.entropy()
                counts[1, c, ws] = len(self.memories)

        print "Total learn time: %1.3f mins" % (tot_learn_time / 60.)
        self.learn_sec = tot_learn_time
        if save_couplings:
            return counts, entropies, couplings
        return counts, entropies

    def sample_from_model(self, trials=None, reshape=False):
        return self.original_spikes.to_windowed(window_size=self.window_size, trials=trials, reshape=reshape)


class BernoulliHomogeneous(SpikeModel):
    """ Bernoulli model of spikes """

    def sample_from_model(self, trials=None, reshape=False):
        """ returns Spikes object of 3d numpy arr of windowed iid Bernouli spike trains:
            (with probabilities = spike rates of each neuron in self)
                X:   T (num trials) x (window_size * N) x  (M - window_size + 1)
                                        binary vector out of a spike time series
            reshape: returns T(M - window_size + 1) x (ws * N) numpy binary vector
        """
        trials = trials or xrange(self.original_spikes.T)
        X = np.zeros(
            (len(trials), self.window_size * self.original_spikes.N, self.original_spikes.M - self.window_size + 1))

        for c, t in enumerate(trials):
            p = self.original_spikes.spikes_arr[t, :, :].mean(axis=1)
            for i in xrange(0, self.original_spikes.M - self.window_size + 1):
                X[c, :, i] = sample_from_bernoulli(p, self.window_size).ravel()

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in xrange(len(trials)):
                for c in xrange(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y

        return Spikes(spikes_arr=X)


class BernoulliInhomogeneous(SpikeModel):
    """ Bernoulli model of spikes """

    def sample_from_model(self, averaging_window_size = 20, trials=None, reshape=False):
        trials = trials or xrange(self.original_spikes.T)
        X = np.zeros(
            (len(trials), self.window_size * self.original_spikes.N, self.original_spikes.M - self.window_size + 1))

        for c, t in enumerate(trials):
            num_neurons = self.original_spikes.N
            num_samples = self.original_spikes.M
            spikes_arr = self.original_spikes.spikes_arr

            ps = []
            for i in xrange(num_neurons):
                ps.append(
                    [spikes_arr[0, i, 0:averaging_window_size].mean()] + [spikes_arr[0, i, (j - 1) * averaging_window_size:j * averaging_window_size].mean() for j
                                                            in xrange(1, num_samples / averaging_window_size)])
            ps = np.array(ps)

            for j in xrange(num_neurons):
                for i in xrange(0, self.original_spikes.M - self.window_size + 1):
                    X[c, j, i] = int(np.random.random() < ps[j, i / averaging_window_size])
                    # sample_from_Bernoulli([ps[j,i/numbins]], 1).ravel()[0]

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in xrange(len(trials)):
                for c in xrange(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y
        return Spikes(spikes_arr=X)



class Shuffled(SpikeModel):
    """ shuffle spikes """

    def sample_from_model(self, trials=None, trial_independence=True, reshape=False):
        """ returns new Spikes object: permutes spikes_arr in time
            trial_independence: diff permutation for each trial """
        idx = np.random.permutation(self.original_spikes.M)
        new_arr = np.zeros(self.original_spikes.spikes_arr.shape)
        for i in xrange(self.original_spikes.T):
            if trial_independence:
                idx = np.random.permutation(self.original_spikes.M)
            arr = self.original_spikes.spikes_arr[i, :, :].copy()
            new_arr[i] = arr[:, idx]
        return Spikes(new_arr).to_windowed(
            window_size=self.window_size, trials=trials)



class Ising(SpikeModel):
    """ Ising / Hopfield model of spikes 
        WARNING:  NOT QUITE WORKING !!! """

    def sample_from_model(self, J=None, theta=None, trials=None, reshape=False):
        """ WARNING: NOT FUNCTIONING PROPERLY I THINK

        returns new spikes object with iid Ising spike trains:
            (with Ising model determined by learning with MPF)
        """

        trials = trials or range(self.original_spikes.T)
        X = np.zeros((len(trials), self.original_spikes.N, self.original_spikes.M))

        learner = Learner(spikes=self.original_spikes)

        no_net = False
        if J is None or theta is None:
            no_net = True

        for c, t in enumerate(trials):
            if no_net:
                learner.learn_from_spikes(window_size=1, trials=[t])
                J = learner.network.J
                theta = learner.network.theta
            X[c, :, :] = sample_from_ising(J, theta, self.original_spikes.M)

        return Spikes(spikes_arr=X)

<<<<<<< HEAD

class DichotomizedGaussian(SpikeModel):
    def sample_from_model(self, trials=None, reshape=False):

        trials = trials or range(self.original_spikes.T)

        spikes_windowed = self.original_spikes.to_windowed(self.window_size, trials)

        X = np.zeros((len(trials), self.original_spikes.N, self.original_spikes.M))

        # statistics
        for c, t in enumerate(trials):
            mu = spikes_windowed.spikes_arr[t, :, :].mean(axis=1)
            cov = np.cov(spikes_windowed.spikes_arr[t, :, :])
            gamma, rho = find_latent_gaussian(mu, cov)

            print "mu ", mu.shape
            print "cov ", cov.shape

            for i in xrange(0, spikes_windowed.M, self.window_size):
                x = sample_from_dichotomized_gaussian(mu, cov, 1, gamma, rho)
                X[c, :, i:i+self.window_size] = x.reshape(self.original_spikes.N, self.window_size)

            if spikes_windowed.M % self.window_size != 0:
                stub = spikes_windowed.M % self.window_size
                x = sample_from_dichotomized_gaussian(mu, cov, 1, gamma, rho)

                X[c, :, spikes_windowed.M - stub : spikes_windowed.M] = \
                    x.reshape(self.original_spikes.N, self.window_size)[:, :stub]

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in xrange(len(trials)):
                for c in xrange(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y

        return Spikes(spikes_arr=X)

class DichotomizedGaussianPoisson(SpikeModel):
    def sample_from_model(self, trials=None, reshape=False):
        """
        [samples,gammas,Lambda,joints2D,cmfs,hists] = DGPoisson(means,Sigma,Nsamples,acc)
          Generates samples from a Multivariate Discretized Gaussian with Poisson marginals
          and specified covariance. Also returns parameters of the fitted DG.

          Inputs:
            means: the means of the input Poissons. Must be a vector with n elements.
            Sigma: The covariance matrix of the input-random variable. The function
              does not check for admissability, i.e. results might be wrong if there
              exists no random variable which has the specified marginals and
              covariance.
            Nsamples: The number of samples to be generated.
            acc: the desired accuracy. If empty or missing, default ist used

          Outputs:
            samples: a matrix of size Nsamples by n, where each row is a sample from
              the DG.
            gammas: the discretization thresholds, as described in the paper. When
              sampling. The k-th dimension of the output random variable is f if e.g.
              supports{k}(1)=f and gammas{k}(f) <= U(k) <= gammas{k}(f+1)
            Lambda: the covariance matrix of the latent Gaussian random variable U
            joints2D: An n by n cell array, where each entry contains the 2
              dimensional joint distribution of  a pair of dimensions of the DG.
            hists: the empirical marginals of the samples returned in "samples"

        Code from the paper: 'Generating spike-trains with specified
        correlations', Macke et al., submitted to Neural Computation

        www.kyb.mpg.de/bethgegroup/code/efficientsampling
        """

        means = []
        Sigma = []
        Nsamples = 1
        accuracy = 1e-10

        # calculate marginal distribution of Poisson
        pmfs, cmfs, supports = poisson_marginals(means, accuracy)


        # find paramters of DG
        gammas, Lambda, joints2D = find_dg_any_marginal(pmfs, Sigma, supports)

        # generate samples
        samples, hists = sample_dg_any_marginal(gammas, Lambda, Nsamples, supports)


