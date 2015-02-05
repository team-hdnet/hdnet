# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import os
import numpy as np
from data import SpkReader

from hdnet.spikes import Spikes

from test_tmppath import TestTmpPath


class TestData(TestTmpPath):

    def setUp(self):
        super(TestData, self).setUp()

    def tearDown(self):
        super(TestData, self).tearDown()

    def test_basic(self):
        spikes = SpkReader.read_spk_files(
            [os.path.join(os.path.dirname(__file__), fn)
             for fn in ['test_data/t00.spk', 'test_data/t02.spk', 'test_data/t04.spk', 'test_data/t08.spk']],
            bin_size=1)
        self.assertTrue(spikes.spikes.shape, (1, 4, 721883))

        spikes = SpkReader.read_spk_folder(
            os.path.join(os.path.dirname(__file__), 'test_data'), bin_size=1)
        self.assertTrue(spikes.spikes.shape, (1, 4, 721883))

# end of source
