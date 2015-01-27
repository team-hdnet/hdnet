# -*- coding: utf-8 -*-
"""
    hdnet.learner
    ~~~~~~~~~~~~~

    Class for learning hopfield network on spikes trains

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np
import os

import hopfield_mpf
from counter import Counter


class Learner(object):
    """ takes spikes and learns a network on windowed patterns
    
    Parameters
        network: Hopfield network
        spikes: 
        window_size: number of contiguous N-bit (N neurons) vectors in time to train on
    """

    def __init__(self, spikes=None, network=None, network_file=None, window_size=1, params=None):
        self.window_size = window_size

        if spikes is not None:
            self.spikes = spikes
            self.spikes_file = spikes.filename
            self.network = hopfield_mpf.HopfieldNetMPF(self.spikes.N * self.window_size)

        if network is not None:
            self.network = network

        if network_file is not None and network is None:
            self.network.loadz(network_file)

        if params is not None:
            self.params = params
        else:
            self.params = {'Mode': 'default'}

    _savevars = ["spikes_file", "window_size"]

    def learn_from_binary(self, X, remove_zeros=False):
        """ trains on M x N matrix X of M N-length binary vects """
        self.network = hopfield_mpf.HopfieldNetMPF(len(X[0]))
        if remove_zeros:
            X_ = X[X.mean(axis=1) != 0., :]  # remove all zeros
            print "Learning %d %d-bit (nonzero) binary patterns, sparsity %.04f..." % (
            X_.shape[0], X_.shape[1], X_.mean())
        else:
            X_ = X
            print "Learning %d %d-bit binary patterns, sparsity %.04f..." % (X_.shape[0], X_.shape[1], X_.mean())
        self.network.learn_all(X_)

    def learn_from_spikes(self, spikes=None, window_size=1, trials=None, remove_zeros=True):
        """ trains network over spikes """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = hopfield_mpf.HopfieldNetMPF(spikes.N * self.window_size)
        X = spikes.to_windowed(window_size=self.window_size, trials=trials, reshape=True)
        self.learn_from_binary(X, remove_zeros=remove_zeros)

    def learn_from_spikes_rot(self, spikes=None, window_size=1, trials=None, remove_zeros=True):
        """ trains network over spikes """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = hopfield_mpf.HopfieldNetMPF(spikes.N * self.window_size)
        c = Counter(save_fp_sequence=True)
        c.chomp_spikes(spikes, window_size=self.window_size, rotate=(spikes.N, self.window_size))
        X = np.array([c.reverse_key(c.fp_list[m]) for m in
                      c.sequence])  # spikes.to_windowed(window_size=self.window_size, trials=trials, reshape=True)
        self.learn_from_binary(X, remove_zeros=remove_zeros)

    def save(self, folder_name='my_learner'):
        """ saves as npz's: network, params, spikes filename """
        if os.path.exists(folder_name):  # replace with Exception
            print "Folder %s exists" % folder_name
            return

        os.mkdir(folder_name)
        self.network.savez(folder_name + '/network')
        d = dict([(v, getattr(self, v)) for v in self._savevars])
        np.savez(folder_name + '/save_vars', **d)
        np.savez(folder_name + "/params", **self.params)

    def load(self, folder_name='my_learner'):
        if not os.path.exists(folder_name):
            print "Folder %s does not exist" % folder_name
            return

        if not hasattr(self, 'network'):
            self.network = hopfield_mpf.HopfieldNetMPF()
        self.network.loadz(folder_name + '/network.npz')

        d = np.load(folder_name + '/save_vars.npz')
        for v in self._savevars:
            setattr(self, v, d[v])

        self.params = dict(np.load(folder_name + '/params.npz'))


# end of source
