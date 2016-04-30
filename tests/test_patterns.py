# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import os
import numpy as np

from hdnet.spikes import Spikes
from hdnet.learner import Learner
from hdnet.patterns import Counter, PatternsHopfield, PatternsRaw
from hdnet.util import hdlog

from test_tmppath import TestTmpPath


class TestPatternsHopfield(TestTmpPath):

    def setUp(self):
        super(TestPatternsHopfield, self).setUp()
        import logging
        logging.disable(level=logging.WARNING)

    def tearDown(self):
        super(TestPatternsHopfield, self).tearDown()

    def test_counter(self):
        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        hdlog.info(spikes._spikes)

        counter = Counter()
        counter.chomp_spikes(spikes)
        hdlog.info(counter._counts)
        self.assertEqual(len(counter), 4)

        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 4)

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 9)

        counter.save(os.path.join(self.TMP_PATH, 'counter'))
        counter2 = Counter.load(os.path.join(self.TMP_PATH, 'counter'))
        self.assertTrue(isinstance(counter2, Counter))
        self.assertEqual(len(counter), len(counter2))

        spikes_arr1 = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
        spikes = Spikes(spikes=spikes_arr1)
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
        spikes = Spikes(spikes=spikes_arr2)
        counter4 = Counter().chomp_spikes(spikes).merge_counts(counter3)
        self.assertEqual(len(counter4.counts.keys()), 5)
        self.assertEqual(len(counter4.patterns), 5)

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        counter = Counter(save_sequence=True)
        counter.chomp_spikes(spikes)
        self.assertEqual(counter._sequence, [0, 1, 0, 2, 3, 4, 1, 5, 4, 6, 2, 0, 6, 2, 2])

        np.random.seed(42)
        spikes_arr = (np.random.randn(5, 10000) < .05).astype(np.int)
        spikes = Spikes(spikes=spikes_arr)
        empirical = PatternsRaw()
        empirical.chomp_spikes(spikes)
        empirical_w2 = PatternsRaw()
        empirical_w2.chomp_spikes(spikes, window_size=2)
        self.assertTrue(np.abs(empirical_w2.entropy() - 2 * empirical.entropy()) < .1)

    def test_patterns_raw(self):
        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        hdlog.info(spikes._spikes)
        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes)
        hdlog.info(patterns._counts)
        self.assertEqual(len(patterns), 4)

        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(patterns), 4)

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        patterns = PatternsRaw()
        patterns.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(patterns), 9)

        patterns.save(os.path.join(self.TMP_PATH, 'raw'))
        patterns2 = PatternsRaw.load(os.path.join(self.TMP_PATH, 'raw'))
        self.assertTrue(isinstance(patterns2, PatternsRaw))
        self.assertEqual(len(patterns), len(patterns2))

    def test_patterns_hopfield(self):
        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)

        patterns = PatternsHopfield(learner=learner)
        patterns.chomp_spikes(spikes)
        # print spikes.spikes
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
        # print spikes.spikes
        
        # print patterns.counts
        self.assertEqual(len(patterns), 4)
        # print "%d fixed-points (entropy H = %1.3f):" % (len(patterns), patterns.entropy())
        # for x in patterns.list_patterns(): print x

        spikes_arr1 = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
        spikes = Spikes(spikes=spikes_arr1)
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)

        # test recording fixed-points
        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes)
        patterns = PatternsHopfield(learner, save_sequence=True)
        patterns.chomp_spikes(spikes)
        self.assertEqual(patterns._sequence, [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        learner = Learner(spikes)
        learner.learn_from_spikes(spikes, window_size=2)
        patterns = PatternsHopfield(learner, save_sequence=True)
        patterns.chomp_spikes(spikes, window_size=2)
        # print patterns.mtas
        # print patterns.sequence
        # for x in patterns.list_patterns(): print x
        # print spikes.spikes
        self.assertEqual(patterns._sequence, [0, 1, 2, 3, 0, 1, 4, 5, 6, 5, 7, 3])
        # self.assertTrue(np.mean(patterns.pattern_to_binary_matrix(1) == [[0, 0], [0, 1], [1, 0]]))
        # self.assertTrue(np.mean(patterns.pattern_to_mta_matrix(1) == [[0, 0], [0, 1], [1, .5]]))
        
        hdlog.info(spikes._spikes)
        hdlog.info(patterns.pattern_to_trial_raster(3))
        # print patterns.pattern_to_mta_matrix(1)
        # print patterns.pattern_to_binary_matrix(1)
        # print patterns.pattern_to_mta_matrix(0)
        # print patterns.pattern_to_binary_matrix(0)


# end of source
