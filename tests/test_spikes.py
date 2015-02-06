# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import os
import numpy as np

from hdnet.spikes import Spikes

from test_tmppath import TestTmpPath


class TestSpikes(TestTmpPath):

    def setUp(self):
        super(TestSpikes, self).setUp()
        import logging
        logging.disable(level=logging.WARNING)

    def tearDown(self):
        super(TestSpikes, self).tearDown()

    def test_basic(self):
        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/tiny_spikes.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        self.assertEqual(spikes._spikes.sum(), 9)
        self.assertEqual(spikes.rasterize(stop=5).sum(), 7)

        spikes.rasterize(save_png_name=os.path.join(self.TMP_PATH, 'spikes'))
        self.assertTrue(os.path.exists(os.path.join(self.TMP_PATH, 'spikes.png')))

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        spikes.rasterize(save_png_name=os.path.join(self.TMP_PATH, 'spikes'))
        self.assertTrue(os.path.exists(os.path.join(self.TMP_PATH, 'spikes.png')))

        file_contents = np.load(os.path.join(os.path.dirname(__file__), 'test_data/spikes_trials.npz'))
        spikes = Spikes(file_contents[file_contents.keys()[0]])
        spikes.restrict_to_most_active_neurons(top_neurons=2)
        self.assertEqual(spikes._N, 2)
        self.assertEqual(len(spikes.idx), 2)

    def test_saving(self):
        spikes = Spikes(spikes=np.array([[1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]]))
        print spikes.spikes

        spikes.save(os.path.join(self.TMP_PATH, 'spikes'))
        spikes2 = Spikes.load(os.path.join(self.TMP_PATH, 'spikes'))
        print spikes2.spikes
        self.assertTrue((spikes.spikes == spikes2.spikes).all())


# end of source
