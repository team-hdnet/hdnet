# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Counter class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil
import numpy as np

from hdnet.spikes import Spikes
from hdnet.counter import Counter


class TestCounter(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        if os.path.exists("counter"):
            shutil.rmtree("counter")
        os.mkdir("counter")

    def tearDown(self):
        if os.path.exists("counter"):
            shutil.rmtree("counter")

    def test_basic(self):
        spikes = Spikes(npz_file='test_data/tiny_spikes.npz')
        print spikes.spikes_arr
        counter = Counter()
        counter.chomp_spikes(spikes)
        print counter.counts
        self.assertEqual(len(counter), 3)

        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 4)

        spikes = Spikes(npz_file='test_data/spikes_trials.npz')
        counter = Counter()
        counter.chomp_spikes(spikes, window_size=3)
        self.assertEqual(len(counter), 9)

        spikes_arr1 = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
        spikes = Spikes(spikes_arr=spikes_arr1)
        counter1 = Counter()
        counter1.chomp_spikes(spikes)
        counter2 = Counter()
        counter2.chomp_spikes(spikes)
        counter2.merge_counts(counter1)
        self.assertEqual(sum(counter2.counts.values()), 6)
        counter3 = counter2 + counter1
        self.assertEqual(counter3, counter2)
        self.assertEqual(sum(counter3.counts.values()), 9)

        spikes_arr2 = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]])
        spikes = Spikes(spikes_arr=spikes_arr2)
        counter4 = Counter().chomp_spikes(spikes).merge_counts(counter3)
        self.assertEqual(len(counter4.counts.keys()), 5)
        self.assertEqual(len(counter4.fp_list), 5)

        spikes = Spikes(npz_file='test_data/spikes_trials.npz')
        counter = Counter(save_fp_sequence=True)
        counter.chomp_spikes(spikes)
        self.assertEqual(counter.sequence, [0, 1, 0, 2, 3, 4, 1, 5, 4, 6, 2, 0, 6, 2, 2])

        spikes_arr = np.random.randn(5, 10000)
        spikes = Spikes(spikes_arr=spikes_arr)
        self.assertTrue(np.abs(spikes.spikes_arr[0, :, :].mean() - .5) < .1)
        emperical = Counter()
        emperical.chomp_spikes(spikes)
        emperical_w2 = Counter()
        emperical_w2.chomp_spikes(spikes, window_size=2)
        self.assertTrue(np.abs(emperical_w2.entropy() - 2 * emperical.entropy()) < 1)
