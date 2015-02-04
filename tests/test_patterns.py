# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Pattern class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import numpy as np

from hdnet.spikes import Spikes
from hdnet.learner import Learner
from hdnet.patterns import Counter, PatternsHopfield, PatternsRaw

from test_tmppath import TestTmpPath


class TestPatternsHopfield(TestTmpPath):

    def setUp(self):
        super(TestPatternsHopfield, self).setUp()

    def tearDown(self):
        super(TestPatternsHopfield, self).tearDown()

    def test_counter(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        print spikes.spikes_arr
        counter = Counter()
        counter.chomp_spikes(spikes)
        print counter._counts
        self.assertEqual(len(counter), 3)

        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 4)

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 9)

        counter.save(os.path.join(self.TMP_PATH, 'counter'))
        counter2 = Counter.load(os.path.join(self.TMP_PATH, 'counter'))
        self.assertTrue(isinstance(counter2, Counter))
        self.assertEqual(len(counter), len(counter2))

        spikes_arr1 = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
        spikes = Spikes(spikes_arr=spikes_arr1)
        counter1 = Counter()
        counter1.chomp_spikes(spikes)
        counter2 = Counter()
        counter2.chomp_spikes(spikes)
        counter2.merge_counts(counter1)
        self.assertEqual(sum(counter2._counts.values()), 6)
        counter3 = counter2 + counter1
        self.assertEqual(counter3, counter2)
        self.assertEqual(sum(counter3.counts.values()), 9)

        spikes_arr2 = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]])
        spikes = Spikes(spikes_arr=spikes_arr2)
        counter4 = Counter().chomp_spikes(spikes).merge_counts(counter3)
        self.assertEqual(len(counter4.counts.keys()), 5)
        self.assertEqual(len(counter4.patterns), 5)

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        counter = Counter(save_sequence=True)
        counter.chomp_spikes(spikes)
        self.assertEqual(counter._sequence, [0, 1, 0, 2, 3, 4, 1, 5, 4, 6, 2, 0, 6, 2, 2])

        spikes_arr = np.random.randn(5, 10000)
        spikes = Spikes(spikes_arr=spikes_arr)
        self.assertTrue(np.abs(spikes.spikes_arr[0, :, :].mean() - .5) < .1)
        empirical = Counter()
        empirical.chomp_spikes(spikes)
        empirical_w2 = Counter()
        empirical_w2.chomp_spikes(spikes, window_size=2)
        self.assertTrue(np.abs(empirical_w2.entropy() - 2 * empirical.entropy()) < 1)

    def test_patterns_raw(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        print spikes.spikes_arr
        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes)
        print patterns._counts
        self.assertEqual(len(patterns), 3)

        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(patterns), 4)

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(patterns), 9)

        patterns.save(os.path.join(self.TMP_PATH, 'raw'))
        counter2 = Counter.load(os.path.join(self.TMP_PATH, 'raw'))
        self.assertTrue(isinstance(counter2, Counter))
        self.assertEqual(len(patterns), len(counter2))

    def test_patterns_hopfield(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)

        patterns = PatternsHopfield(learner=learner)
        patterns.chomp_spikes(spikes)
        # print spikes.spikes_arr
        self.assertEqual(len(patterns), 3)
        # print "%d fixed-points (entropy H = %1.3f):" % (len(patterns), patterns.entropy())
        # print map(patterns.pattern_for_key, patterns.counts.keys())

        patterns.save(os.path.join(self.TMP_PATH, 'patterns'))
        patterns2 = PatternsHopfield.load(os.path.join(self.TMP_PATH, 'patterns'))
        self.assertTrue(isinstance(patterns2, PatternsHopfield))
        self.assertEqual(len(patterns2), 3)
        self.assertEqual(len(patterns2.mtas), 3)
        self.assertEqual(len(patterns2.mtas_raw), 3)

        learner.learn_from_spikes(spikes, window_size=3)
        patterns = PatternsHopfield(learner=learner)
        patterns.chomp_spikes(spikes, window_size=3)
        # print spikes.spikes_arr
        
        # print patterns.counts
        self.assertEqual(len(patterns), 4)
        # print "%d fixed-points (entropy H = %1.3f):" % (len(patterns), patterns.entropy())
        # for x in patterns.list_patterns(): print x

        spikes_arr1 = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
        spikes = Spikes(spikes_arr=spikes_arr1)
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)

        # test recording fixed-points
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)
        patterns = PatternsHopfield(learner, save_sequence=True)
        patterns.chomp_spikes(spikes)
        self.assertEqual(patterns._sequence, [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes, window_size=2)
        patterns = PatternsHopfield(learner, save_sequence=True)
        patterns.chomp_spikes(spikes, window_size=2)
        # print patterns.mtas
        # print patterns.sequence
        # for x in patterns.list_patterns(): print x
        # print spikes.spikes_arr
        self.assertEqual(patterns._sequence, [0, 1, 2, 3, 0, 1, 4, 5, 6, 5, 7, 3])
        # self.assertTrue(np.mean(patterns.pattern_to_binary_matrix(1) == [[0, 0], [0, 1], [1, 0]]))
        # self.assertTrue(np.mean(patterns.pattern_to_mta_matrix(1) == [[0, 0], [0, 1], [1, .5]]))
        
        print spikes.spikes_arr
        print patterns.pattern_to_trial_raster(3)
        # print patterns.pattern_to_mta_matrix(1)
        # print patterns.pattern_to_binary_matrix(1)
        # print patterns.pattern_to_mta_matrix(0)
        # print patterns.pattern_to_binary_matrix(0)


# end of source
