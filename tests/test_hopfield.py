# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import unittest
from time import time as now
import numpy as np

from hdnet.hopfield import HopfieldNet, HopfieldNetMPF
from hdnet.util import hdlog

class TestHopfield(unittest.TestCase):
    def setUp(self):
        import logging
        logging.disable(level=logging.WARNING)

    def test_basic(self):
        MPF = HopfieldNetMPF(N=3)
        MPF._J = np.array([[0, -1, 1], [-1, 0, 2], [1, 2, 0]])

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
        np.random.seed(42)

        N = 300
        M = N / (4 * np.log(N))  # theoretical max for OPR [McEliece et al, 87]
        t = now()
        OPR = HopfieldNet(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        OPR.learn_all(data)

        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
        # recall = (data == OPR.hopfield_binary_dynamics(data, model='OPR')).all(1).mean()
        # recall = OPR.exact_recalled(data, model='OPR')
        hdlog.info("OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t))
        self.assertTrue(recall > .8)

        M = 50
        OPR = HopfieldNet(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        OPR.learn_all(data)
        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
        # recall = (data == OPR.hopfield_binary_dynamics(data, model='OPR')).all(1).mean()
        # recall = OPR.exact_recalled(data)
        hdlog.info("OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t))
        self.assertTrue(recall < .5)

        MPF = HopfieldNetMPF(N)
        MPF.learn_all(data)
        recall = MPF.exact_recalled(data)
        hdlog.info("MPF Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t))
        self.assertEqual(recall, 1)

        # store 90 memories in 64-bit neurons
        N = 64
        M = 90
        t = now()
        MPF = HopfieldNetMPF(N)
        data = (np.random.random((M, N)) < .5).astype('int')
        MPF.learn_all(data)
        recall = MPF.exact_recalled(data)
        hdlog.info("MPF Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t))
        self.assertEqual(recall, 1)
        OPR = HopfieldNet(N)
        OPR.learn_all(data)
        recall = (data == OPR.hopfield_binary_dynamics(data)).all(1).mean()
        # recall = OPR.exact_recalled(data)
        hdlog.info("OPR Performance (%d/%d): %1.2f in %1.2f s" % (M, N, 100 * recall, now() - t))
        self.assertTrue(recall < .01)

# end of source
