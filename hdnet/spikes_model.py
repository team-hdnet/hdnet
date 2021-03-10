# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.spikes_model
    ~~~~~~~~~~~~~~~~~~

    Null-models for spikes' statistics.

"""

from __future__ import print_function

import numpy as np
from time import time as now
from hdnet.stimulus import Stimulus

from hdnet.spikes import Spikes
from hdnet.patterns import PatternsRaw, PatternsHopfield
from hdnet.learner import Learner
from hdnet.sampling import sample_from_bernoulli, sample_from_ising_gibbs, sample_from_dichotomized_gaussian, \
    find_latent_gaussian, poisson_marginals, find_dg_any_marginal, sample_dg_any_marginal
from hdnet.util import Restoreable, hdlog


class SpikeModel(Restoreable, object):
    """
    Generic model of spikes (and stimulus).

    Parameters
    ----------
    spikes : Type, optional
        Description (default None)
    stimulus : Type, optional
        Description (default None)
    window_size : int, optional
        Description (default 1)
    learner : Type, optional
        Description (default None)

    Parameters
    spikes: spikes to model
    stimulus: corresp stimulus if existent
    window_size: length of time window in binary bins

    Returns
    -------
    Value : Type
        Description
    """
    _SAVE_ATTRIBUTES_V1 = ['_window_size', '_learn_time']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'SpikeModel'
    _INTERNAL_OBJECTS = zip([Spikes, Spikes, Spikes, PatternsRaw, PatternsHopfield, Stimulus, Learner],
                            ['_original_spikes', '_sample_spikes', '_hopfield_spikes',
                              '_raw_patterns', '_hopfield_patterns', '_stimulus', '_learner'],
                             ['spikes_original', 'spikes_sample', 'spikes_hopfield',
                              'patterns_raw', 'patterns_hopfield', 'stimulus', 'learner'])

    def __init__(self, spikes=None, stimulus=None, window_size=1, learner=None):
        object.__init__(self)
        Restoreable.__init__(self)

        self._stimulus = stimulus
        self._window_size = window_size
        self._learner = learner or None
        self._original_spikes = spikes
        self._learn_time = None
        self._sample_spikes = None
        self._raw_patterns = None
        self._hopfield_patterns = None
        self._hopfield_spikes = None

    @property
    def stimulus(self):
        """
        Getting the stimulus object associated with the spikes_model

        Returns
        -------
        self._stimulus : Stimulus
            The Stimulus Object mainly consisting of a numpy array which
            could give you all the stimuli frames
        """
        return self._stimulus

    @property
    def window_size(self):
        """
        Getting the Window Size of the spikes_model

        Returns
        -------
        self._window_size : Integer
            The Window size of the concerned spikes_model
        """
        return self._window_size

    @property
    def learner(self):
        """
        Getting the learner object which shall be used to fit this model

        Returns
        -------
        self._learner : Learner
            This object could be used to train a network on the Spikes based on
            some Training Rule.
        """
        return self._learner

    @property
    def original_spikes(self):
        """
        Getting the original unaltered spikes

        Returns
        -------
        self._original_spikes : Spikes
            The original unaltered spikes train
        """
        return self._original_spikes

    @property
    def learn_time(self):
        """
        Returns the total learning/fitting time taken by the latest run learning
        rule to fit the model over the concerned window_sizes

        Returns
        -------
        self._learn_time: float
            The Amount of time taken to fit the latest run learning rule
        """
        return self._learn_time

    @property
    def sample_spikes(self):
        """
        Returns the sample spikes associated with the spikes_model

        Returns
        -------
        self._sample_spikes : Spikes
            The Sample spikes associated with the model
        """
        return self._sample_spikes

    @property
    def raw_patterns(self):
        """
        Returns the raw patterns associated with the model

        Returns
        -------
        self._raw_patterns : PatternsRaw
            Contains a catalogue of Binary Patterns and their prevalence in the
            raw spikes train.
        """
        return self._raw_patterns

    @property
    def hopfield_patterns(self):
        """
        Returns the object of PatternsHopfield class associated with the SpikeModel

        Returns
        -------
        self._hopfield_patterns : PatternsHopfield
            A catalogue of Hopfield Fixed points of binary vectors alongwith their
            prevalence
        """
        return self._hopfield_patterns

    @property
    def hopfield_spikes(self):
        """
        Returns the spikes over which we'll apply the Hopfield Network Dynamics

        Returns
        -------
        self._hopfield_spikes : Spikes
            These are the spikes, over which we'll be applying the Hopfield
            Net Dynamics to reach a fixed point.
        """
        return self._hopfield_spikes

    def fit(self, trials=None, remove_zeros=True, reshape=False):
        """
        This helps us fit Hopfield Networks to a sample from the original spikes
        data

        Parameters
        ----------True,
        trials : Numpy Array, optional
            Consists of all the trials which we want to have while training
        remove_zeros : bool, optional
            Remove all 0 training patterns (default True)(As these will vanish while Applying
            Hopfield Dynamics and have no significance)
        reshape : bool, optional
            Set to true if you want your 3D Spikes Array squeezed to 2D (default False)

        Returns
        -------
        void
        """
        self._sample_spikes = self.sample_from_model(trials=trials, reshape=reshape)
        self._learner = Learner(spikes=self._sample_spikes)
        self._learner.learn_from_spikes(remove_zeros=remove_zeros)

    def chomp(self):
        """
        Used to chomp/count all occurences of all patterns that occur in the spikes

        Returns
        -------
        void
        """
        hdlog.info("Chomping samples from model")
        self._raw_patterns = PatternsRaw(save_sequence=True)
        self._raw_patterns.chomp_spikes(spikes=self._sample_spikes)
        hdlog.info("Raw: %d-bit, %d patterns" % (
            self._sample_spikes.N, len(self._raw_patterns)))

        hdlog.info("Chomping dynamics (from network learned on the samples) applied to samples")
        self._hopfield_patterns = PatternsHopfield(learner=self._learner, save_sequence=True)
        self._hopfield_patterns.chomp_spikes(spikes=self._sample_spikes)
        hdlog.info("Hopfield: %d-bit, %d patterns" % (
            self._sample_spikes.N, len(self._hopfield_patterns)))

        # print "Before dynamics:"
        # print self.sample_spikes.spikes
        # print "Applied dynamics:"
        self._hopfield_spikes = self._hopfield_patterns.apply_dynamics(spikes=self._sample_spikes, reshape=True)
        # ma_err = np.abs(self.sample_spikes.spikes - hop_model_spikes).mean()
        #        print hop_model_spikes
        # print "Mean prediction: %1.4f/1.0 (vs guess zero: %1.4f)" % (
        #    (1 - ma_err), 1 - np.abs(self.sample_spikes.spikes).mean())
        # # distortion
        # self.sample_spikes

    def distinct_patterns_over_windows(self, window_sizes=None, trials=None, save_couplings=False, remove_zeros=False):
        """
        Returns tuple: counts, entropies [, couplings]
        counts, entropies: arrays of size 2 x T x WSizes
        (0: empirical from model sample, 1: dynamics from learned model on sample)

        Parameters
        ----------
        window_sizes : Numpy Array, optional
            All Window Sizes over which you'd want to operate (default None)
        trials : Numpy Array, optional
            All trials for which you'd want to calculate entropy etc.(default None)
        save_couplings : bool, optional
            Denotes whether or not you want a copy of the Weight Matrix
            from all trained HopfieldNets to be returned to you (default False)
        remove_zeros : bool, optional
            Whether or not you want to remove the zeros in order to ease training (default False)

        Returns
        -------
        counts, entropies: Numpy arrays of size 2 x T x WSizes
        (0: empirical from model sample, 1: dynamics from learned model on sample)

        """
        if window_sizes is None:
            window_sizes = [1]
        trials = trials or range(self._original_spikes.T)
        counts = np.zeros((2, len(trials), len(window_sizes)))
        entropies = np.zeros((2, len(trials), len(window_sizes)))
        selfi = self        #Buffer Variable to do stuff, as we don't want to change self
        tot_learn_time = 0

        for ws, window_size in enumerate(window_sizes):
            for c, trial in enumerate(trials):
                hdlog.info("Trial %d | ws %d" % (trial, window_size))

                selfi._window_size = window_size

                t = now()
                selfi.fit(trials=[trial], remove_zeros=remove_zeros)
                diff = now() - t
                hdlog.info("[%1.3f min]" % (diff / 60.))
                tot_learn_time += diff

                if save_couplings:
                    couplings[trial].append(self._learner.network.J.copy())

                selfi.chomp()
                entropies[0, c, ws] = selfi._raw_patterns.entropy()
                counts[0, c, ws] = len(selfi._raw_patterns)
                entropies[1, c, ws] = selfi._hopfield_patterns.entropy()
                counts[1, c, ws] = len(selfi._hopfield_patterns)

        hdlog.info("Total learn time: %1.3f mins" % (tot_learn_time / 60.))
        self._learn_time = tot_learn_time
        if save_couplings:
            return counts, entropies, couplings
        return counts, entropies

    def entropy(self):
        pass

    def sample_from_model(self, trials=None, reshape=False):
        """
        Get a sample from SpikesData appropriately windowed and restricted to only
        the given trials, and/or reshaped if you want them to

        Parameters
        ----------
        trials : Numpy Array, optional
            The Trials which you want to still keep in your model
        reshape : bool, optional
            If you want to reduce your 3D Spikes Numpy array to 2D

        Returns
        -------
        (windowed)self._original_spikes : Spikes
            The Original Spikes Train provided to us are now windowed,
            reshaped, restricted to only certain trials
        """
        return self._original_spikes.to_windowed(window_size=self._window_size, trials=trials, reshape=reshape)

    def save(self, folder_name='spikes_model', overwrite=False):
        """
        saves as npz's: network, params, spikes file_name

        Parameters
        ----------
        folder_name : str, optional
            Description (default 'spikes_model')
        overwrite: bool, optional
            Overwrite flag, whether to overwrite existing files (default False)

        Returns
        -------
        void
        """
        super(SpikeModel, self)._save(
            'spikes_model.npz', self._SAVE_ATTRIBUTES_V1, self._SAVE_VERSION,
            has_internal=True, folder_name=folder_name, internal_objects=self._INTERNAL_OBJECTS,
            overwrite=overwrite)

    @classmethod
    def load(cls, folder_name='spikes_model', load_extra=False):
        """
        Used to load a previously saved spikes_model into Memory

        Parameters
        ----------
        folder_name : str, optional
            The File Name/Directory address where the file is present(default 'spikes_model')
        load_extra : bool, optional
            Attribute to be later used by loader() (default False)

        Returns
        -------
        The loaded object of SpikesModel Class
        """
        return super(SpikeModel, cls)._load('spikes_model.npz', has_internal=True,
                                            folder_name=folder_name,
                                            internal_objects=cls._INTERNAL_OBJECTS,
                                            load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        # internal function to load v1 file format
        hdlog.debug('Loading SpikeModel, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<SpikeModel: {s}, window size {ws}>'.\
            format(s=repr(self.original_spikes), ws=self.window_size)


class BernoulliHomogeneous(SpikeModel):
    """
    Bernoulli model of spikes, homogeneous.
    """

    def sample_from_model(self, trials=None, reshape=False):
        """
        Returns Spikes object of 3d numpy arr of windowed iid Bernouli spike trains:
        (with probabilities = spike rates of each neuron in self at trial t)
        X:   T (num trials) x (window_size * N) x  (M - window_size + 1)
        binary vector out of a spike time series
        reshape: returns T(M - window_size + 1) x (ws * N) numpy binary vector

        Parameters
        ----------
        trials : Numpy Array, optional
            The trials which you want to keep in your Spikes Sample (default None)
        reshape : bool, optional
            Whether or not to reshape the Spikes sample to being 2D(default False)

        Returns
        -------
        An object of Spikes Class, with BernoulliHomogeneous distribution of spikes
        """
        trials = trials or range(self._original_spikes.T)
        X = np.zeros(
            (len(trials), self._window_size * self._original_spikes.N, self._original_spikes.M - self._window_size + 1))

        sample_spikes = self._original_spikes.to_windowed(trials=trials, window_size=self._window_size)
        for c, t in enumerate(trials):
            p = sample_spikes.spikes[c, :, :].mean(axis=1)
            X[c, :, :] = sample_from_bernoulli(p, self._original_spikes.M - self._window_size + 1)

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in range(len(trials)):
                for c in range(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y

        return Spikes(spikes=X)


class BernoulliInhomogeneous(SpikeModel):
    """ Bernoulli model of spikes, inhomogeneous (i.e. varying rate over time). """

    def sample_from_model(self, averaging_window_size = 20, trials=None, reshape=False):
        """
        Returns Spikes which are distributed as per an inhomogeneous Bernoulli distribution
        i.e. The probability of activation varies over different sections of spikes but is
        same for those within a particular window only

        Parameters
        ----------
        averaging_window_size : int, optional
            The Window Size, the averaged probability of whose activation will
            give us the overall probability for that particular window (default 20)
        trials : Type, optional
            The trials for which you want to construct your Spikes (default None)
        reshape : bool, optional
            Whether or not you want to squeeze your spikes to 2D(default False)

        Returns
        -------
        An Object of Spikes Class containing the inhomogeneous Bernoulli Distribution
        """
        trials = trials or range(self._original_spikes.T)
        X = np.zeros(
            (len(trials), self._window_size * self._original_spikes.N, self._original_spikes.M - self._window_size + 1))

        for c, t in enumerate(trials):
            num_neurons = self._original_spikes.N
            num_samples = self._original_spikes.M
            spikes = self._original_spikes.spikes

            ps = []
            for i in range(num_neurons):
                ps.append(
                    [spikes[0, i, (j - 1) * averaging_window_size:j * averaging_window_size].mean() for j
                                                            in range(1, num_samples / averaging_window_size)])
            ps = np.array(ps)

            for j in range(num_neurons):
                for i in range(0, self._original_spikes.M - self._window_size + 1):
                    X[c, j, i] = int(np.random.random() < ps[j, i / averaging_window_size])
                    # sample_from_Bernoulli([ps[j,i/numbins]], 1).ravel()[0]

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in range(len(trials)):
                for c in range(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y
        return Spikes(spikes=X)


class Shuffled(SpikeModel):
    """
    Shuffled spikes
    """

    def sample_from_model(self, trials=None, trial_independence=True, reshape=False):
        """
        returns new Spikes object: permutes spikes in time
        trial_independence: diff permutation for each trial

        Parameters
        ----------
        trials : Numpy Array, optional
            containing the trials we want to sample (default None)
        trial_independence : bool, optional
            Deciding if we'll have a different permutation for each trial (default True)
        reshape : bool, optional
            Whether or not we want the Spikes Sample squeezed from 3D to 2D(default False)

        Returns
        -------
        An Object of Spikes Class containing a shuffled sample of the original spikes
        """
        idx = np.random.permutation(self._original_spikes.M)
        new_arr = np.zeros(self._original_spikes.spikes.shape)
        for i in range(self._original_spikes.T):
            if trial_independence:
                idx = np.random.permutation(self._original_spikes.M)
            arr = self._original_spikes.spikes[i, :, :].copy()
            new_arr[i] = arr[:, idx]
        return Spikes(new_arr).to_windowed(
            window_size=self._window_size, trials=trials)


class Ising(SpikeModel):
    """
    Class modeling the Ising / Hopfield model of spikes

    """

    def sample_from_model(self, J=None, theta=None, trials=None, reshape=False):
        """
        Returns new spikes object with iid Ising spike trains:
        (with Ising model determined by learning with MPF)

        Parameters
        ----------
        J : Numpy Array, optional
            The Weight Matrix for Ising Model (default None)
        theta : Numpy Array, optional
            The threshold Vector for Ising Model (default None)
        trials : Type, optional
            The trials which you'd want to change (default None)
        reshape : bool, optional
            Whether or not you'd want to squeeze the spike train to 2D (default False)

        Returns
        -------
        An object of Spikes Class containing the Ising Model of Spikes
        """

        trials = trials or range(self._original_spikes.T)
        X = np.zeros((len(trials), self._original_spikes.N, self._original_spikes.M))

        learner = Learner(spikes=self._original_spikes)

        no_net = False
        if J is None or theta is None:
            no_net = True

        for c, t in enumerate(trials):
            if no_net:
                learner.learn_from_spikes(window_size=1, trials=[t])
                J = learner._network.J
                theta = learner._network.theta
            X[c, :, :] = sample_from_ising_gibbs(J, theta, self._original_spikes.M)

        return Spikes(spikes=X)


class DichotomizedGaussian(SpikeModel):
    """
    Class modeling the dichotomized Gaussian model of spikes.
    """
    def sample_from_model(self, trials=None, reshape=False):
        """
        Getting a Sample from the DichotomizedGaussian Model, i.e. For each neuron
        the activation values shall adhere to a certain Gaussian Distribution which
        shall be created using the original_spikes

        Parameters
        ----------
        trials : Numpy Array, optional
            The Trials which are to be considered (default None)
        reshape : bool, optional
            Whether or not we are to squeeze the Spikes to 2D (default False)

        Returns
        -------
        An Object of Spikes Class
        """
        trials = trials or range(self._original_spikes.T)
        spikes_windowed = self._original_spikes.to_windowed(self._window_size, trials)
        X = np.zeros((len(trials), self._original_spikes.N, self._original_spikes.M))

        # statistics
        for c, t in enumerate(trials):
            bin_means = spikes_windowed.spikes[t, :, :].mean(axis=1)
            bin_cov = np.cov(spikes_windowed.spikes[t, :, :])
            gauss_means, gauss_cov = find_latent_gaussian(bin_means, bin_cov)

            for i in range(0, spikes_windowed.M, self._window_size):
                x = sample_from_dichotomized_gaussian(bin_means, bin_cov, 1, gauss_means, gauss_cov)
                X[c, :, i:i+self._window_size] = x.reshape(self._original_spikes.N, self._window_size)

            if spikes_windowed.M % self._window_size != 0:
                stub = spikes_windowed.M % self._window_size
                x = sample_from_dichotomized_gaussian(bin_means, bin_cov, 1, gauss_means, gauss_cov)

                X[c, :, spikes_windowed.M - stub : spikes_windowed.M] = \
                    x.reshape(self._original_spikes.N, self._window_size)[:, :stub]

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in range(len(trials)):
                for c in range(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y

        return Spikes(spikes=X)


class DichotomizedGaussianPoisson(SpikeModel):
    """
    Class modeling the dichotomized Gaussian model of spikes, with Poisson marginals.
    """

    def sample_from_model(self, trials=None, reshape=False):
        """
        Get a Spikes Sample Modelled with DichotomizedGaussian with Poisson marginals

        Parameters
        ----------
        trials : Type, optional
            Description (default None)
        reshape : bool, optional
            Description (default False)

        Returns
        -------
        Value : Type
            Description
        """

        trials = trials or range(self._original_spikes.T)
        spikes_windowed = self._original_spikes.to_windowed(self._window_size, trials)
        X = np.zeros((len(trials), self._original_spikes.N, self._original_spikes.M))

        # statistics
        for c, t in enumerate(trials):
            bin_means = spikes_windowed.spikes[t, :, :].mean(axis=1)
            bin_cov = np.cov(spikes_windowed.spikes[t, :, :])

            # calculate marginal distribution of Poisson
            pmfs, cmfs, supports = poisson_marginals(bin_means)

            # find paramters of DG
            gauss_means, gauss_cov, joints = find_dg_any_marginal(pmfs, bin_cov, supports)

            # generate samples
            samples, hists = sample_dg_any_marginal(gauss_means, gauss_cov, self._original_spikes.M, supports)

            X[c, :, :] = samples.T

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in range(len(trials)):
                for c in range(X.shape[2]):
                    Y[tot, :] = X[t, :, c]
                    tot += 1
            return Y

        return Spikes(spikes=X)


# end of source
