# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Stimulus class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil

from hdnet.stimulus import Stimulus


class TestStimulus(unittest.TestCase):

    def test_basic(self):

        stimulus = Stimulus(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_stimulus.npz'))
        self.assertEqual(stimulus.stimulus_arr.shape, (2, 4, 4))

        # stimulus = Stimulus(h5_file='test_data/shortmovie.h5')
        # self.assertEqual(stimulus.stimulus_arr.shape, (4500, 32, 32))
        # self.assertEqual(stimulus.M, 4500)
        # self.assertEqual(stimulus.X, (32, 32))
        # 
        # snap = stimulus.snapshot()
        # stimulus.snapshot(0, 20, 'test')
