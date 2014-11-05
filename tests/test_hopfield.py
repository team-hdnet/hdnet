# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Tests for Hopfield class

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import unittest
import shutil
from time import time as now
import numpy as np

from hdnet.hopfield import HopfieldNet
from hdnet.hopfield_mpf import HopfieldNetMPF


class TestHopfield(unittest.TestCase):

    def test_basic(self):
        MPF = HopfieldNetMPF(N=3)
        MPF.J = np.array([[0, -1, 1], [-1, 0, 2], [1, 2, 0]])

        x = np.array([0, 1, 0])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([0, 0, 0])), 0.)

        x = np.array([1, 0, 0])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([0, 0, 0])), 0.)

        x = np.array([1, 1, 0])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([0, 0, 0])), 0.)

        x = np.array([0, 0, 1])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([1, 1, 1])), 0.)

        x = np.array([1, 0, 1])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([1, 1, 1])), 0.)

        x = np.array([1, 1, 1])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([0, 1, 1])), 0.)

        x = np.array([0, 1, 1])
        y = MPF(x, converge=False)
        self.assertEqual(np.linalg.norm(y - np.array([0, 1, 1])), 0.)

        self.assertEqual(MPF.energy(np.array([1, 1, 0])), 1)
        self.assertEqual(MPF.energy(np.array([0, 0, 0])), 0)
        self.assertEqual(MPF.energy(np.array([1, 0, 0])), 0)
        self.assertEqual(MPF.energy(np.array([0, 1, 0])), 0)
        self.assertEqual(MPF.energy(np.array([0, 0, 1])), 0)
        self.assertEqual(MPF.energy(np.array([1, 0, 1])), -1)
        self.assertEqual(MPF.energy(np.array([1, 1, 1])), -2)
        self.assertEqual(MPF.energy(np.array([0, 1, 1])), -2)

    def test_learning(self):
        # OPR learning (Hopfield original rule)
        N = 300
        M = N / (4 * np.log(N))  # theoretical max for OPR [McEliece et al, 87]
        t = now()
        OPR = HopfieldNet(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        OPR.learn_all(data)

        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
#        recall = (data == OPR.hopfield_binary_dynamics(data, model='OPR')).all(1).mean()
#        recall = OPR.exact_recalled(data, model='OPR')
        print "OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t)
        self.assertTrue(recall > .8)

        M = 50
        OPR = HopfieldNet(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        OPR.learn_all(data)
        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
#        recall = (data == OPR.hopfield_binary_dynamics(data, model='OPR')).all(1).mean()
        # recall = OPR.exact_recalled(data)
        print "OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t)
        self.assertTrue(recall < .5)

        MPF = HopfieldNetMPF(N)
        MPF.learn_all(data)
        recall = MPF.exact_recalled(data)
        print "MPF Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t)
        self.assertEqual(recall, 1)

        # store 90 memories in 64-bit neurons
        N = 64
        M = 90
        t = now()
        MPF = HopfieldNetMPF(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        MPF.learn_all(data)
        recall = MPF.exact_recalled(data)
        print "MPF Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t)
        self.assertEqual(recall, 1)
        OPR = HopfieldNet(N)
        OPR.learn_all(data)
        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
#        recall = OPR.exact_recalled(data)
        print "OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t)
        self.assertTrue(recall < .01)
