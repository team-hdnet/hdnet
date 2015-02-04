# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Spikes class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import numpy as np

from hdnet.spikes import Spikes

from test_tmppath import TestTmpPath


class TestSpikes(TestTmpPath):

    def setUp(self):
        super(TestSpikes, self).setUp()

    def tearDown(self):
        super(TestSpikes, self).tearDown()

    def test_basic(self):
        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))

        self.assertEqual(spikes._spikes_arr.sum(), 9)
        self.assertEqual(spikes.rasterize(stop=5).sum(), 7)

        spikes.rasterize(save_png_name=os.path.join(self.TMP_PATH, 'spikes'))
        self.assertTrue(os.path.exists(os.path.join(self.TMP_PATH, 'spikes.png')))

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes.rasterize(save_png_name=os.path.join(self.TMP_PATH, 'spikes'))
        self.assertTrue(os.path.exists(os.path.join(self.TMP_PATH, 'spikes.png')))

        spikes = Spikes(spk_files=[os.path.join(os.path.dirname(__file__), fn) for fn in
            ['test_data/t00.spk', 'test_data/t02.spk', 'test_data/t04.spk',
            'test_data/t08.spk' ]], bin_size=1)
        self.assertTrue(spikes._spikes_arr.shape, (1, 4, 721883))

        spikes = Spikes(spk_folder=os.path.join(os.path.dirname(__file__), 'test_data'), bin_size=1)
        self.assertTrue(spikes._spikes_arr.shape, (1, 4, 721883))

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        print spikes._spikes_arr

        spikes = Spikes(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes.restrict_to_most_active_neurons(top_neurons=2)
        self.assertEqual(spikes._N, 2)
        self.assertEqual(len(spikes.idx), 2)

        spikes = Spikes(spikes_arr=np.array([[1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]]))
        print spikes._spikes_arr

        spikes.save(os.path.join(self.TMP_PATH, 'spikes'))
        spikes2 = Spikes.load(os.path.join(self.TMP_PATH, 'spikes'))
        self.assertTrue((spikes.spikes_arr == spikes2.spikes_arr).all())


# end of source
