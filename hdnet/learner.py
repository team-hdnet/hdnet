# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.learner
    ~~~~~~~~~~~~~

    Class for learning hopfield network on spikes trains

"""

import numpy as np

from hdnet.hopfield import HopfieldNetMPF, HopfieldNet
from hdnet.patterns import Counter
from hdnet.util import Restoreable, hdlog


class Learner(Restoreable, object):
    """
    Takes spikes and learns a network on windowed patterns.

    Parameters
    ----------
    spikes : :class:`.Spikes`, optional
        Spikes class instance to use (default None)
    network : :class:`.HopfieldNet`, optional
        HopfieldNetwork class instance to use (default None)
    network_file : str, optional
        File name of Hopfield network to load (default None)
    window_size : int, optional
        Size of window in bins (default 1)
    params : dict, optional
        Dictionary of optional parameters (default None)

    Returns
    -------
    learner : :class:`.Learner`
        Instance of class :class:`.Learner`
    """
    _SAVE_ATTRIBUTES_V1 = ['_spikes_file', '_window_size', '_params']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Learner'
    _INTERNAL_OBJECTS = zip([HopfieldNet], ['_network'], ['network'])

    def __init__(self, spikes=None, network=None, network_file=None, window_size=1, params=None):
        object.__init__(self)
        Restoreable.__init__(self)

        self._window_size = window_size

        if spikes is not None:
            self._spikes = spikes
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
        """
        Getter for hopfield network of this learner.
        
        Returns
        -------
        network : :class:`.HopfieldNet`
            Network of this learner
        """
        return self._network

    @network.setter
    def network(self, value):
        """
        Setter for setting the network of this learner.
        
        Parameters
        ----------
        value : :class:`.HopfieldNet`
            HopfieldNet network to set
        
        Returns
        -------
        Nothing
        """
        self._network = value

    @property
    def params(self):
        """
        Getter for parameters of learner.
        
        Returns
        -------
        parameters : dict
            Parameters of learner
        """
        return self._params

    @params.setter
    def params(self, value):
        """
        Setter for parameters of learner.
        
        Parameters
        ----------
        value : dict
            New dictionary to assign
        
        Returns
        -------
        Nothing
        """
        self._params = value

    @property
    def window_size(self):
        """
        Getter for window size.
        
        Returns
        -------
        value : int
            Value of window size
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        """
        Setter for window size.
        
        Parameters
        ----------
        value : int
            Value of window size to set
        
        Returns
        -------
        Nothing
        """
        self._window_size = value

    @property
    def spikes(self):
        """
        Setter for spikes.
        
        Returns
        -------
        value : :class:`.Spikes`
            Instance of :class:`.Spikes` class
        """
        return self._spikes

    @spikes.setter
    def spikes(self, value):
        """
        Setter for spikes.
        
        Parameters
        ----------
        value : :class:`.Spikes`
            Instance of :class:`.Spikes` class
            to assign
        
        Returns
        -------
        Nothing
        """
        self._spikes = value

    @property
    def spikes_file(self):
        """
        Getter for spikes file.
        
        Returns
        -------
        file : str
            File name of spikes file
        """
        return self._spikes_file

    @spikes_file.setter
    def spikes_file(self, value):
        """
        Setter for spikes file.
        
        Parameters
        ----------
        value : str
            Value of spikes file to set
        
        Returns
        -------
        Nothing
        """
        self._spikes_file = value

    def learn_from_binary(self, X, remove_zeros=False, disp=False):
        """
        Trains on M x N matrix X of M N-length binary vects

        Parameters
        ----------
        X : numpy array
            (M, N)-dim array of binary input patterns of length N,
            where N is the number of nodes in the network
        remove_zeros : bool, optional
            Flag whether to remove vectors from X in which
            all entries are 0 (default True)
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)

        Returns
        -------
        Nothing
        """
        self.network = HopfieldNetMPF(len(X[0]))
        if remove_zeros:
            X_ = X[X.mean(axis=1) != 0., :]  # remove all zeros
            hdlog.info("Learning %d %d-bit (nonzero) binary patterns, sparsity %.04f..." % (
                X_.shape[0], X_.shape[1], X_.mean()))
        else:
            X_ = X
            hdlog.info("Learning %d %d-bit binary patterns, sparsity %.04f..." % (X_.shape[0], X_.shape[1], X_.mean()))
        self.network.learn_all(X_, disp=disp)

    def learn_from_spikes(self, spikes=None, window_size=1, trials=None, remove_zeros=True, disp=False):
        """
        Trains network over spikes contained in instance of :class:`.Spikes` class.
        
        Parameters
        ----------
        spikes : :class:`.Spikes`, optional
            Instance of Spikes class (default None)
        window_size : int, optional
            Window size to use (default 1)
        trials : Type, optional
            Description (default None)
        remove_zeros : bool, optional
            Flag whether to remove windows in which
            all entries are 0 (default True)
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)
        
        Returns
        -------
        Nothing
        """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = HopfieldNetMPF(spikes.N * self.window_size)
        X = spikes.to_windowed(window_size=self.window_size, trials=trials, reshape=True)
        self.learn_from_binary(X, remove_zeros=remove_zeros, disp=disp)

    def learn_from_spikes_rot(self, spikes=None, window_size=1, trials=None, remove_zeros=True, disp=False):
        """
        Trains network over spikes contained in instance of :class:`.Spikes` class,
        removes windows that are identical modulo a rotation along the first axis.
        
        Parameters
        ----------
        spikes : :class:`.Spikes`, optional
            Instance of Spikes class (default None)
        window_size : int, optional
            Window size to use (default 1)
        trials : Type, optional
            Description (default None)
        remove_zeros : bool, optional
            Flag whether to remove windows in which
            all entries are 0 (default True)
        disp : bool, optional
            Display scipy L-BFGS-B output (default False)

        Returns
        -------
        Nothing
        """
        spikes = spikes or self.spikes
        trials = trials or range(spikes.T)
        self.window_size = window_size
        self.network = HopfieldNetMPF(spikes.N * self.window_size)
        c = Counter(save_sequence=True)
        c.chomp_spikes(spikes, window_size=self.window_size, rotate=(spikes.N, self.window_size))
        X = np.array([Counter.pattern_for_key(c.patterns[m]) for m in c.sequence])
        self.learn_from_binary(X, remove_zeros=remove_zeros, disp=disp)

    def save(self, folder_name='learner'):
        """
        Saves Learner to file. Also saves the contained instance of
        :class:`.HopfieldNet`.

        Parameters
        ----------
        folder_name : str, optional
            Folder name name to save Learner to (default 'learner')

        Returns
        -------
        Nothing
        """
        super(Learner, self)._save(
            'learner.npz', self._SAVE_ATTRIBUTES_V1, self._SAVE_VERSION,
            has_internal=True, folder_name=folder_name, internal_objects=self._INTERNAL_OBJECTS)

    @classmethod
    def load(cls, folder_name='learner', load_extra=False):
        """
        Loads Learner from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            learner = Learner.load('folder_name')

        Parameters
        ----------
        folder_name : str, optional
            Folder name name to load Learner from (default 'learner')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        learner : :class:`.Learner`
            Instance of :class:`.Learner` if loaded, `None` upon error
        """
        return super(Learner, cls)._load('learner.npz', has_internal=True,
                                         folder_name=folder_name,
                                         internal_objects=cls._INTERNAL_OBJECTS,
                                         load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        # internal function to load v1 file format
        hdlog.debug('Loading Learner, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<Learner: on network with {n} nodes>'.format(n=self.network.N if self.network.N is not None else 0)

# end of source
