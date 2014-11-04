# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Spikes class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil
import numpy as np

from hdnet.spikes import Spikes


class TestSpikes(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        if os.path.exists("spikes"):
            shutil.rmtree("spikes")
        os.mkdir("spikes")

    def tearDown(self):
        if os.path.exists("spikes"):
            shutil.rmtree("spikes")

    def test_basic(self):
        spikes = Spikes(npz_file='test_data/tiny_spikes.npz')

        self.assertEqual(spikes.spikes_arr.sum(), 9)
        
        self.assertEqual(spikes.rasterize(stop=5).sum(), 7)

        spikes.rasterize(save_png_name='spikes/spikes')

        self.assertTrue(os.path.exists("spikes/spikes.png"))

        spikes = Spikes(npz_file='test_data/spikes_trials.npz')

        spikes.rasterize(save_png_name='spikes/spikes')
        self.assertTrue(os.path.exists("spikes/spikes.png"))

        spikes = Spikes(spk_files=['test_data/t00.spk', 'test_data/t02.spk', 'test_data/t04.spk', 
            'test_data/t08.spk', ], bin_size=1)
        self.assertTrue(spikes.spikes_arr.shape, (1, 4, 721883))

        spikes = Spikes(spk_folder='test_data', bin_size=1)
        self.assertTrue(spikes.spikes_arr.shape, (1, 4, 721883))

        spikes = Spikes(npz_file='test_data/tiny_spikes.npz')

        print spikes.spikes_arr
        print spikes.to_windowed_bernoulli(window_size=2)

        spikes = Spikes(npz_file='test_data/spikes_trials.npz')
        spikes.restrict_to_most_active_neurons(top_neurons=2)
        self.assertEqual(spikes.N, 2)
        self.assertEqual(len(spikes.idx), 2)

        spikes = Spikes(spikes_arr=np.array([[1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]]))
        print spikes.spikes_arr
        spikes_ising = spikes.to_ising_spikes()
        print spikes_ising.spikes_arr
