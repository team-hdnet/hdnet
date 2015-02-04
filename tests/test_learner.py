# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Learner class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os

from hdnet.spikes import Spikes
from hdnet.learner import Learner

from test_tmppath import TestTmpPath


class TestLearner(TestTmpPath):

    def setUp(self):
        super(TestLearner, self).setUp()

    def tearDown(self):
        super(TestLearner, self).tearDown()

    def test_basic(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        learner = Learner(spikes)
        self.assertEqual(learner.spikes.N, 3)

        learner.learn_from_spikes()
        self.assertTrue(learner.network.J.mean() != 0.)

        learner.learn_from_spikes(spikes)
        self.assertTrue(learner.network.J.mean() != 0.)

        learner.learn_from_spikes(spikes, window_size=3)
        self.assertTrue(learner.network.J.mean() != 0.)
        self.assertTrue(learner.network.J.shape == (9, 9))

        learner.params['hi'] = 'chris'
        learner.save(os.path.join(self.TMP_PATH, 'learner'))
        learner = Learner()
        learner.load(os.path.join(self.TMP_PATH, 'learner'))
        self.assertEqual(learner.params['hi'], 'chris')
        self.assertEqual(learner.spikes_file, os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        self.assertEqual(learner.window_size, 3)
        self.assertTrue(learner.network.J.mean() != 0.)
        self.assertTrue(learner.network.J.shape == (9, 9))

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        learner = Learner(spikes)
        learner.learn_from_spikes()


# end of source
