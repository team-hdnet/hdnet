# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Learner class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil

from hdnet.spikes import Spikes
from hdnet.learner import Learner


class TestLearner(unittest.TestCase):

    TMPPATH = '/tmp/hdnettest'

    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        if os.path.exists(self.TMPPATH):
            shutil.rmtree(self.TMPPATH)
        os.mkdir(self.TMPPATH)

    def tearDown(self):
        if os.path.exists(self.TMPPATH):
            shutil.rmtree(self.TMPPATH)

    def test_basic(self):
        spikes = Spikes(npz_file='test_data/tiny_spikes.npz')
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
        learner.savez(self.TMPPATH+'/learner')
        learner = Learner()
        learner.loadz(self.TMPPATH+'/learner')
        self.assertEqual(learner.params['hi'], 'chris')
        self.assertEqual(learner.spikes_file, 'test_data/tiny_spikes.npz')
        self.assertEqual(learner.window_size, 3)
        self.assertTrue(learner.network.J.mean() != 0.)
        self.assertTrue(learner.network.J.shape == (9, 9))

        spikes = Spikes(npz_file='test_data/spikes_trials.npz')
        learner = Learner(spikes)
        learner.learn_from_spikes()
