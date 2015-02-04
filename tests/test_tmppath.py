# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests template for saving/loading features

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""


import os
import unittest
import shutil


class TestTmpPath(unittest.TestCase):

    TMP_PATH = '/tmp/hdnettest'

    def setUp(self):
        if os.path.exists(self.TMP_PATH):
            shutil.rmtree(self.TMP_PATH)
        os.mkdir(self.TMP_PATH)

    def tearDown(self):
        if os.path.exists(self.TMP_PATH):
            shutil.rmtree(self.TMP_PATH)


# end of source
