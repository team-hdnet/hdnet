# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

import unittest
import numpy as np

from hdnet.sampling import sample_from_prob_vector, sample_from_bernoulli, sample_from_ising_gibbs, \
    sample_from_dichotomized_gaussian


class TestSampling(unittest.TestCase):

    def setUp(self):
        import logging
        logging.disable(level=logging.WARNING)

    def test_basic(self):
        np.random.seed(42)

        p = [.1, .4, .5]
        self.assertTrue(sample_from_prob_vector(p) > -1)
        self.assertTrue(sample_from_prob_vector(p, 100).mean() > 1)
        sample_from_bernoulli(p, 10)

        p = [.8, .05, .05, .05, .05]
        sample_from_bernoulli(p, 10)
        sample_from_bernoulli(p)

        p = [.5, .4, .1]
        self.assertTrue(sample_from_prob_vector(p, 100).mean() < 1)

        p = np.random.random(100)
        p /= p.sum()

        a = np.arange(0, len(p))
        exp_state = np.dot(a, p)

        self.assertTrue(np.abs(sample_from_prob_vector(p, 1000).mean() - exp_state) < 5)

    def test_ising(self):
        np.random.seed(42)
        J = np.random.random((4, 4)) * 3.
        J -= np.diag(J.diagonal())
        J += J.T
        theta = np.zeros(4)
        expected_means = np.array([ 0.2, 0.2, 0.2, 0.2])
        self.assertTrue(np.all(sample_from_ising_gibbs(
            J, theta, 10000, 4 * 100, 4 * 10).mean(axis = 1) - expected_means < 1e-2))

    def test_dichotomous(self):
        np.random.seed(42)

        mu = np.array([.4, .3])
        cov = np.array([[.24, .1], [.1, .21]])

        X = sample_from_dichotomized_gaussian(mu, cov, 1000)[0]
        X.mean(axis=1)

        self.assertTrue(np.sum(X.mean(axis=1) - np.array([ 0.396,  0.307])) < 1e-5)
        self.assertTrue(np.sum(X.var(axis=1) - np.array([ 0.239184,  0.212751])) < 1e-5)


# end of source
