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

from hopfield import HopfieldNetMPF, HopfieldNet
from patterns import Counter
from util import Restoreable, hdlog 


class Learner(Restoreable, object):
    """ takes spikes and learns a network on windowed patterns
    
    Parameters
        network: Hopfield network
        spikes: 
        window_size: number of contiguous N-bit (N neurons) vectors in time to train on
    """
    _SAVE_ATTRIBUTES_V1 = ['_spikes_file', '_window_size', '_params']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Learner'

    def __init__(self, spikes=None, network=None, network_file=None, window_size=1, params=None):
        object.__init__(self)
        Restoreable.__init__(self)

        self._window_size = window_size

        if spikes is not None:
            self._spikes = spikes
            self._spikes_file = spikes.filename
            self._network = HopfieldNetMPF(self._spikes.N * self._window_size)

        if network is not None:
            self._network = network

        if network_file is not None and network is None:
            self._network = HopfieldNet.load(network_file)

        if params is not None:
            self._params = params
        else:
            self._params = {'Mode': 'default'}

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self._window_size = value

    @property
    def spikes(self):
        return self._spikes

    @spikes.setter
    def spikes(self, value):
        self._spikes = value

    @property
    def spikes_file(self):
        return self._spikes_file

    @spikes_file.setter
    def spikes_file(self, value):
        self._spikes_file = value

    def learn_from_binary(self, X, remove_zeros=False):
        """ trains on M x N matrix X of M N-length binary vects """
        self.network = HopfieldNetMPF(len(X[0]))
        if remove_zeros:
            X_ = X[X.mean(axis=1) != 0., :]  # remove all zeros
            hdlog.info("Learning %d %d-bit (nonzero) binary patterns, sparsity %.04f..." % (
                X_.shape[0], X_.shape[1], X_.mean()))
        else:
            X_ = X
            hdlog.info("Learning %d %d-bit binary patterns, sparsity %.04f..." % (X_.shape[0], X_.shape[1], X_.mean()))
        self.network.learn_all(X_)

    def learn_from_spikes(self, spikes=None, window_size=1, trials=None, remove_zeros=True):
        """ trains network over spikes """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = HopfieldNetMPF(spikes.N * self.window_size)
        X = spikes.to_windowed(window_size=self.window_size, trials=trials, reshape=True)
        self.learn_from_binary(X, remove_zeros=remove_zeros)

    def learn_from_spikes_rot(self, spikes=None, window_size=1, trials=None, remove_zeros=True):
        """ trains network over spikes """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = HopfieldNetMPF(spikes.N * self.window_size)
        c = Counter(save_sequence=True)
        c.chomp_spikes(spikes, window_size=self.window_size, rotate=(spikes.N, self.window_size))
        X = np.array([Counter.pattern_for_key(c.patterns[m]) for m in c.sequence])
        self.learn_from_binary(X, remove_zeros=remove_zeros)

    def save(self, folder_name='learner'):
        """ saves as npz's: network, params, spikes filename """
        if os.path.exists(folder_name):  # replace with Exception
            hdlog.error("Folder %s exists, cannot save!" % folder_name)
            return

        os.mkdir(folder_name)
        self.network.save(os.path.join(folder_name, 'network.npz'))
        super(Learner, self)._save(os.path.join(folder_name, 'learner.npz'),
                                  self._SAVE_ATTRIBUTES_V1, self._SAVE_VERSION)

    @classmethod
    def load(cls, folder_name='learner', load_extra=False):
        # TODO: document
        if not os.path.exists(folder_name):
            hdlog.error("Folder %s does not exist!" % folder_name)
            return
        return super(Learner, cls)._load(
            filename=os.path.join(folder_name, 'learner.npz'), load_extra=load_extra).\
             _load_network(os.path.join(folder_name, 'network.npz'))

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('Loading Learner, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    def _load_network(self, file_name):
        self.network = HopfieldNet.load(file_name)
        return self

    # representation

    def __repr__(self):
        return '<Learner: on network with {n} nodes>'.format(n=self.network.N if self.network.N is not None else 0)

# end of source
