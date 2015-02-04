# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details


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
