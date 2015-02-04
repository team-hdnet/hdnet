# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import os
import unittest

from hdnet.stimulus import Stimulus


class TestStimulus(unittest.TestCase):

    def test_basic(self):

        stimulus = Stimulus(npz_file=os.path.join(os.path.dirname(__file__), 'test_data/tiny_stimulus.npz'))
        self.assertEqual(stimulus._stimulus_arr.shape, (2, 4, 4))

        # stimulus = Stimulus(h5_file='test_data/shortmovie.h5')
        # self.assertEqual(stimulus.stimulus_arr.shape, (4500, 32, 32))
        # self.assertEqual(stimulus.M, 4500)
        # self.assertEqual(stimulus.X, (32, 32))
        # 
        # snap = stimulus.snapshot()
        # stimulus.snapshot(0, 20, 'test')

# end of source
