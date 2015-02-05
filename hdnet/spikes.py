# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.spikes
    ~~~~~~~~~~~~

    Spikes class handling multi-neuron , multi-trial spike trains.

"""

__version__ = "0.1"

import os
import numpy as np
from util import hdlog, Restoreable
from visualization import save_matrix_whole_canvas


class Spikes(Restoreable, object):
    """
    Class for handling binary time-series datasets.

    Creates a :class:`.Spikes` class from a binary 2d or 3d numpy array ``spikes``.

    If the array is 2-dimensional, the first dimension is assumed to represent
    neurons and the second one to represent bins.

    If the array is 3-dimensional, the first dimension is assumed to represent
    trials, the second one to represent neurons and the third one to represent bins.

    Parameters
    ----------
    spikes : numpy array
        Raw binned spikes
    bin_size : int, optional
        Number of subsequent bins to merge (default 1)
    preprocess : bool, optional
        If True makes data binary (Heaviside), if False
        leaves data untouched (default True)

    Returns
    -------
    spikes : instance of :class:`Spikes` class

    """
    _SAVE_ATTRIBUTES_V1 = ['_spikes', '_T', '_N', '_M']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Spikes'

    def __init__(self, spikes=None, bin_size=1, preprocess=True):
        object.__init__(self)
        Restoreable.__init__(self)
        self._spikes = np.atleast_2d(spikes)

        spikes_shape = self._spikes.shape
        if len(spikes_shape) == 2:  # single trial
            self._spikes = self._spikes.reshape((1, spikes_shape[0], spikes_shape[1]))

        self._T = self._spikes.shape[0]
        self._N = self._spikes.shape[1]
        self._M = self._spikes.shape[2]

        if preprocess:
            self._preprocess()

    def _preprocess(self):
        # make binary
        self._spikes = np.double((np.sign(self._spikes) + 1) // 2)

    @property
    def spikes(self):
        """
        Returns underlying numpy array representing spikes, with
        dimensions organized as follows (trials, neurons, bins)
        
        Returns
        -------
        spikes : 3d numpy array
        """
        return self._spikes

    @property
    def num_neurons(self):
        """
        Returns number of neurons represented by this class.

        Returns
        -------
        number of neurons : int
        """
        return self._N

    @property
    def N(self):
        """
        Returns number of neurons represented by this class,
        shortcut for :meth:`num_neurons`.
        
        Returns
        -------
        number of neurons : int
        """
        return self._N

    @property
    def num_bins(self):
        """
        Returns number of bins represented by this class.

        Returns
        -------
        number of bins : int
        """
        return self._M


    @property
    def M(self):
        """
        Returns number of bins represented by this class,
        shortcut for :meth:`num_bins`.
        
        Returns
        -------
        number of bins : int
        """
        return self._M

    @property
    def num_trials(self):
        """
        Returns number of trials represented by this class.

        Returns
        -------
        number of trials : int
        """
        return self._T

    @property
    def T(self):
        """
        Returns number of trials represented by this class,
        shortcut for :meth:`num_trials`.

        Returns
        -------
        number of trials : int
        """
        return self._T

    def restrict_to_most_active_neurons(self, top_neurons=None, copy=False):
        """
        Restricts the selection to the :attr:`top_neurons` most active
        (does not make a copy) if top_neurons is None: sorts the spike_arr
        
        Parameters
        ----------
        top_neurons : int, optional
            number of most active neurons to choose, if None all are chosen (default None)
        copy : bool, optional
            if True returns a new Spikes class with selected neurons, if False
            the changes are made in place, dropping all but the selected neurons (default False)

        Returns
        -------
        spikes : Spikes
            an instance of :class:`.Spikes` class
        """
        self._N = top_neurons or self._N
        activity = self._spikes[:, :, :].mean(axis=0).mean(axis=1)
        idx = activity.argsort()
        self.idx = idx[-self._N:]
        self.mean_activities = activity[self.idx]
        restricted = self._spikes[:, idx[-self._N:], :]
        if copy:
            return Spikes(spikes=restricted)
        else:
            self._spikes = restricted
            return self

    def to_windowed(self, window_size=1, trials=None, reshape=False):
        """
        returns new Spikes object of 3d numpy arr of windowed spike trains:
        X:   T (num trials) x (window_size * N) x  (M - window_size + 1)
        binary vector out of a spike time series
        reshape: returns T(M - window_size + 1) x (ws * N) numpy binary vector
        
        Parameters
        ----------
        window_size : int, optional
            Description (default 1)
        trials : Type, optional
            Description (default None)
        reshape : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        trials = trials or range(self._T)
        X = np.zeros((len(trials), window_size * self._N, self._M - window_size + 1))
        for c, t in enumerate(trials):
            for i in xrange(0, self._M - window_size + 1):
                X[c, :, i] = self._spikes[t, :, i:window_size + i].ravel()

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in xrange(len(trials)):
                for j in xrange(X.shape[2]):
                    Y[tot, :] = X[t, :, j]
                    tot += 1
            return Y
        return Spikes(spikes=X)

    def rasterize(self, trials=None, start=0, stop=None, save_png_name=None):
        """
        return *new* (copied) numpy array of size (TN x M)
        trials: e.g. [1, 5, 6], None is all
        save_png_name: if not None then only saves
        
        Parameters
        ----------
        trials : Type, optional
            Description (default None)
        start : int, optional
            Description (default 0)
        stop : Type, optional
            Description (default None)
        save_png_name : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        stop = stop or self._M
        trials = trials or range(self._T)
        sub_spikes_arr = self._spikes[trials, :, start:stop]

        if save_png_name is not None:
            save_matrix_whole_canvas(sub_spikes_arr.reshape((len(trials) * self._N, stop - start)),
                                     save_png_name + '.png', cmap='gray')
        else:
            return sub_spikes_arr.copy().reshape((len(trials) * self._N, stop - start))

    def covariance(self, trials=None, start=0, stop=None, save_png_name=None):
        """
        return *new* numpy array of size (T x N x N) which is covariance matrix betwn neurons
        trials: e.g. [0, 1, 5, 6], None is all
        save_png_name: if not None then only saves
        
        Parameters
        ----------
        trials : Type, optional
            Description (default None)
        start : int, optional
            Description (default 0)
        stop : Type, optional
            Description (default None)
        save_png_name : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        stop = stop or self._M
        trials = trials or range(self._T)
        sub_spikes_arr = self._spikes[trials, :, start:stop]

        new_arr = np.zeros((len(trials), self._N, self._N))
        for t, trial in enumerate(trials):
            new_arr[t] = np.cov(sub_spikes_arr[trial])
        new_arr /= new_arr.max()

        if save_png_name is not None:
            new_arr = new_arr.reshape(len(trials) * self._N, self._N)
            save_matrix_whole_canvas(new_arr, save_png_name + '.png', cmap='gray')
        else:
            return new_arr

    # i/o

    def save(self, file_name='spikes', extra=None):
        """
        save as numpy array .npz file
        
        Parameters
        ----------
        file_name : str, optional
            Description (default 'spikes')
        extra : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        return super(Spikes, self)._save(file_name=file_name,
                                         attributes=self._SAVE_ATTRIBUTES_V1, version=self._SAVE_VERSION,
                                         extra=extra)

    @classmethod
    def load(cls, file_name='spikes', load_extra=False):
        """
        Missing documentation
        
        Parameters
        ----------
        file_name : str, optional
            Description (default 'spikes')
        load_extra : bool, optional
            Description (default False)
        
        Returns
        -------
        Value : Type
            Description
        """
        return super(Spikes, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('Loading Spikes, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<Spikes: {n} neurons, {m} bins, {t} trials>'.format(n=self._N, m=self._M, t=self._T)


# end of source
