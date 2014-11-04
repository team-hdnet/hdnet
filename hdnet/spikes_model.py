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
from stimulus import Stimulus
from learner import Learner


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

    def fit(self, trials=None, remove_zeros=True):
        self.sample_spikes = self.sample_from_model(trials=trials)
        self.learner = Learner(spikes=self.sample_spikes)
        self.learner.learn_from_spikes(remove_zeros=remove_zeros)

    def chomp(self):
#        print "Chomping samples from model"
        self.emperical = Counter()
        self.emperical.chomp_spikes(spikes=self.sample_spikes)
        print "%d-bit (emperical): %d patterns (H = %1.3f)" % (self.sample_spikes.N, len(self.emperical), self.emperical.entropy())

#        print "Chomping dynamics (from network learned on the samples) applied to samples"
        self.memories = Patterns(learner=self.learner)
        self.memories.chomp_spikes(spikes=self.sample_spikes)
        print "%d-bit (hopnet): %d memories (H = %1.3f)" % (self.sample_spikes.N, len(self.memories), self.memories.entropy())

        # print "Before dynamics:"
        # print self.sample_spikes.spikes_arr
        # print "Applied dynamics:"
        hop_model_spikes = self.memories.apply_dynamics(spikes=self.sample_spikes, reshape=True)
        ma_err = np.abs(self.sample_spikes.spikes_arr - hop_model_spikes).mean()
#        print hop_model_spikes
        print "Mean prediction: %1.4f/1.0 (vs guess zero: %1.4f)" % ((1-ma_err), 1-np.abs(self.sample_spikes.spikes_arr).mean())
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

    def sample_from_model(self, trials=None):
        return self.original_spikes.to_windowed(window_size=self.window_size, trials=trials)


class Bernoulli(SpikeModel):
    """ Bernoulli model of spikes """

    def sample_from_model(self, trials=None):
        return self.original_spikes.to_windowed_bernoulli(window_size=self.window_size, trials=trials)


class Shuffled(SpikeModel):
    """ shuffle spikes """

    def sample_from_model(self, trials=None, trial_independence=True):
        return self.original_spikes.shuffle(trial_independence=trial_independence).to_windowed(window_size=self.window_size, trials=trials)


class Ising(SpikeModel):
    """ Ising / Hopfield model of spikes 
        WARNING:  NOT QUITE WORKING !!! """

    def sample_from_model(self, trials=None):
        return self.original_spikes.to_ising_spikes(window_size=self.window_size, trials=trials)



