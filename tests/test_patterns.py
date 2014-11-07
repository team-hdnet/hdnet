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
from hdnet.patterns import Patterns
from hdnet.counter import Counter

from test_tmppath import TestTmpPath


class TestPatterns(TestTmpPath):

    def setUp(self):
        super(TestPatterns, self).setUp()

    def tearDown(self):
        super(TestPatterns, self).tearDown()

    def test_basic(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)

        patterns = Patterns(learner=learner)
        patterns.chomp_spikes(spikes)
        # print spikes.spikes_arr
        self.assertEqual(len(patterns), 3)
        # print "%d fixed-points (entropy H = %1.3f):" % (len(patterns), patterns.entropy())
        # print map(patterns.reverse_key, patterns.counts.keys())

        patterns.save(os.path.join(self.TMP_PATH, 'patterns'))
        patterns.load(os.path.join(self.TMP_PATH, 'patterns'))
        self.assertEqual(len(patterns), 3)

        learner.learn_from_spikes(spikes, window_size=3)
        patterns = Patterns(learner=learner)
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

        emperical = Counter()
        emperical.chomp_spikes(spikes)
        patterns = Patterns(learner, emperical)
        self.assertEqual(len(patterns), 2)

        # test recording fixed-points
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)
        patterns = Patterns(learner, save_fp_sequence=True)
        patterns.chomp_spikes(spikes)
        self.assertEqual(patterns.sequence, [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes, window_size=2)
        patterns = Patterns(learner, save_fp_sequence=True)
        patterns.chomp_spikes(spikes, window_size=2)
#        print patterns.stas
        # print patterns.sequence
        # for x in patterns.list_patterns(): print x
        # print spikes.spikes_arr
        self.assertEqual(patterns.sequence, [0, 1, 2, 3, 0, 1, 4, 5, 6, 5, 7, 3])
        self.assertTrue((patterns.fp_to_binary_matrix(1) == [[0, 0], [0, 1], [1, 0]]).all())
        self.assertTrue((patterns.fp_to_sta_matrix(1) == [[0, 0], [0, 1], [1, .5]]).all())
        
        print spikes.spikes_arr
        print patterns.fp_to_trial_raster(3)
        # print patterns.fp_to_sta_matrix(1)
        # print patterns.fp_to_binary_matrix(1)
        # print patterns.fp_to_sta_matrix(0)
        # print patterns.fp_to_binary_matrix(0)
