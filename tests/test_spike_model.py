# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for spike models

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil
import numpy as np

from hdnet.counter import Counter
from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, Shuffled, BernoulliHomogeneous, BernoulliInhomogeneous, DichotomizedGaussian


class TestSpikeModel(unittest.TestCase):

    def test_basic(self):
        # spikes = Spikes(npz_file='test_data/tiny_spikes.npz')
        # spike_model = SpikeModel(spikes=spikes)
        # spike_model.fit()
        # spike_model.chomp()

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spike_model = SpikeModel(spikes=spikes)
        spike_model.fit(remove_zeros=True)
        spike_model.chomp()

        spike_model.fit(remove_zeros=False)
        spike_model.chomp()

        wss = [1, 2]
        counts, entropies = spike_model.distinct_patterns_over_windows(wss, remove_zeros=False)

        bernoulli_model = BernoulliHomogeneous(spikes=spikes)
        bernoulli_model.fit()
        bernoulli_model.chomp()
        
        shuffle_model = Shuffled(spikes=spikes)
        shuffle_model.fit()
        shuffle_model.chomp()

        bernoulli_model = BernoulliHomogeneous(spikes=spikes)
        bernoulli_model.fit()
        bernoulli_model.chomp()
        
        wss = [1, 2, 3]
        counts, entropies = bernoulli_model.distinct_patterns_over_windows(wss)
        
        # sanity check on large Bernoulli example
        spikes_arr = np.random.randn(4, 10000)
        spikes = Spikes(spikes_arr=spikes_arr)
        
        bernoulli_model = BernoulliHomogeneous(spikes=spikes)
        
        wss = [1, 2, 3]
        counts, entropies = bernoulli_model.distinct_patterns_over_windows(wss)
        
        # self.assertTrue(np.abs((entropies[0, 0] / np.array(wss)).mean() - spikes.N) < .1)

        bernoulli_inhom_model = BernoulliInhomogeneous(spikes=spikes)
        bernoulli_inhom_model.fit()
        bernoulli_inhom_model.chomp()

        dichotomized_gaussian = DichotomizedGaussian(spikes=spikes)
        dichotomized_gaussian.sample_from_model()

        dichotomized_gaussian.sample_from_model(window_size=42)

