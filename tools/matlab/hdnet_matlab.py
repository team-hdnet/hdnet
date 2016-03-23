# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

# Matlab wrapper for hdnet

import numpy as np
from hdnet.spikes import Spikes
from hdnet.learner import Learner
from hdnet.patterns import PatternsHopfield

def _convert_2d_array_matlab(X, n):
    return X.reshape((n, X.size / n), order = 'F')   

def fit_ising_matlab(n, X):
    X = _convert_2d_array_matlab(np.array(X).astype(np.int), n)
    spikes = Spikes(spikes = X)
    learner = Learner(spikes)
    learner.learn_from_spikes(spikes, window_size = 1)
    J = learner.network.J
    theta = learner.network.theta
    return J.ravel().tolist(), theta.ravel().tolist()

# end of source
