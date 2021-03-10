# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.spikes
    ~~~~~~~~~~~~

    Spikes class handling multi-neuron , multi-trial spike trains.

"""

from __future__ import print_function

import numpy as np

from hdnet.util import hdlog, Restoreable
from hdnet.visualization import save_matrix_whole_canvas


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
    bin_size : float, optional
        Bin size in seconds (default None)
    preprocess : bool, optional
        If True makes data binary (Heaviside), if False
        leaves data untouched (default False)

    Returns
    -------
    spikes : instance of :class:`.Spikes` class

    """
    _SAVE_ATTRIBUTES_V1 = ['_spikes', '_T', '_N', '_M', '_restricted', '_bin_size']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Spikes'

    def __init__(self, spikes=None, bin_size=None, preprocess=False):
        object.__init__(self)
        Restoreable.__init__(self)
        self._spikes = np.atleast_2d(spikes)

        spikes_shape = self._spikes.shape
        if len(spikes_shape) == 2:  # single trial
            spikes3d = np.zeros((1, spikes_shape[0], spikes_shape[1]))
            spikes3d[0] = self._spikes
            self._spikes = spikes3d

        self._T = self._spikes.shape[0]
        self._N = self._spikes.shape[1]
        self._M = self._spikes.shape[2]

        self._bin_size = bin_size
        self._restricted = None

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
    def bin_size(self):
        """
        Returns the bin size in seconds of the data set.

        Returns
        -------
        bin_size : float
        """
        return self._bin_size

    @bin_size.setter
    def bin_size(self, value):
        """
        Sets the bin size (in seconds, so that e.g. 0.001 corresponds
        to 1ms bins), used to calculate firing rates and in other
        time-based operations.

        Parameters
        ----------
        bin_size : float
            bin size in milliseconds

        Returns
        -------
        Nothing
        """
        self._bin_size = value

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

    def flatten_trials(self):
        """ Concatenates all trials into one long trial

        Parameters
        ----------
        None

        Returns
        -------
        spikes : Spikes
            an instance of :class:`.Spikes` class
        """
        tmp_arr = np.swapaxes(self.spikes, 0, 1)
        return Spikes(spikes=tmp_arr.reshape((tmp_arr.shape[0], tmp_arr.shape[1] * tmp_arr.shape[2])), preprocess=False)

    @property
    def restricted_neurons_indices(self):
        """
        Returns a list of the current neuron indices in the
        original data set after restriction. This will return
        `None` unless the data set has been restricted with
        :meth:`restrict_to` or :meth:`restrict_to_most_active_neurons`.

        Returns
        -------
        indices : list of int
        """
        return self._restricted

    def restrict_to_indices(self, indices, copy=False):
        """
        Restricts the spike data to the neurons with the given `indices`
        in the data set.
        To get the indices of the neurons in the original data set, call
        the method :meth:`restricted_neurons_indices`.
        Note: in the default setting this function does not make a copy but
        drops the unselected neurons from the data set.

        Parameters
        ----------
        indices : 1d list or numpy array
            list of (0-based) indices of neurons to select
        copy : bool, optional
            if True returns a new Spikes class with selected neurons, if False
            the changes are made in place, dropping all but the selected
            neurons (default False)

        Returns
        -------
        spikes : Spikes
            an instance of :class:`.Spikes` class
        """
        restricted = self._spikes[:, indices, :]
        if copy:
            spikes = Spikes(spikes=restricted)
            spikes._restricted = indices
            return spikes
        else:
            self._N = len(indices)
            self._spikes = restricted
            self._restricted = indices
            return self

    def restrict_to_most_active_neurons(self, top_neurons=None, copy=False):
        """
        Restricts the spike data to the number of `top_neurons` most active
        neurons in the data set.
        The neurons are sorted by activity in increasing order.
        To get the indices of the most active neurons in the original data
        set, call the method :meth:`restricted_neurons_indices`.
        Note: in the default setting this function does not make a copy but
        drops the less active neurons from the data set.

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
        activity = self._spikes[:, :, :].mean(axis=0).mean(axis=2)
        idx = activity.argsort()
        if top_neurons is None:
            top_neurons = self._N
        return self.restrict_to_indices(idx[:top_neurons], copy=copy)

    def mean_activity(self):
        """
        Computes the mean activities of all cells (measured in the mean
        number of spikes per bin) in the data set. Multiply with `1./bin_size`
        (`bin_size` in seconds) to obtain firing rates in Hz.
        You can obtain a sorted list of neuron indices by activities by calling
        `mean_activity.argsort()` on the returned numpy array.

        Returns
        -------
        mean_activity : 1d numpy array
            Mean activities of all cells (measured in mean number of spikes per bin)
        """
        return self._spikes[:, :, :].mean(axis=0).mean(axis=1)

    def mean_activity_hz(self):
        """
        Computes the mean activities of all cells in Hz.
        Only works if `bin_size` has been set during creation or via :meth:`bin_size`.
        You can obtain a sorted list of neuron indices by activities by calling
        `mean_activity.argsort()` on the returned numpy array.

        Returns
        -------
        mean_activity : 1d numpy array
            Mean activities of all cells in Hz
        """
        return self._spikes[:, :, :].mean(axis=0).mean(axis=1) / \
               ((self._M / 1000.) * (1. / self._bin_size))

    def trials_average(self, trials=None):
        """
        Computes the average activity over all trials in the data set.

        Parameters
        ----------
        trials : 1d list or numpy array
            list of (0-based) indices of trials to include, if `None`
            all trials are used (default None)

        Returns
        -------
        trials_average : 2d numpy array
            mean activity over trials
        """
        if trials is None:
            return self._spikes[:, :, :].mean(axis=0)
        else:
            return self._spikes[trials, :, :].mean(axis=0)

    def to_windowed(self, window_size=1, trials=None, reshape=False):
        """
        Computes windowed version of spike trains, using a sliding window.

        Returns new Spikes object of 3d numpy arr of windowed spike trains:
        T (num trials) x (window_size * N) x  (M - window_size + 1)
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
        X = np.zeros((len(trials), window_size * self._N, self._M - window_size + 1), dtype = np.int)
        for c, t in enumerate(trials):
            for i in range(0, self._M - window_size + 1):
                X[c, :, i] = self._spikes[t, :, i:window_size + i].ravel()

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]), dtype = np.int)
            tot = 0
            for t in range(len(trials)):
                for j in range(X.shape[2]):
                    Y[tot, :] = X[t, :, j]
                    tot += 1
            return Y
        return Spikes(spikes=X)

    def rasterize(self, trials=None, start=0, stop=None, save_png_name=None):
        """
        Returns *new* (copied) numpy array of size (TN x M)
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
        sub_spikes = self._spikes[trials, :, start:stop]

        if save_png_name is not None:
            save_matrix_whole_canvas(sub_spikes.reshape((len(trials) * self._N, stop - start)),
                                     save_png_name + '.png', cmap='gray')
        else:
            return sub_spikes.reshape((len(trials) * self._N, stop - start))

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
        sub_spikes = self._spikes[trials, :, start:stop]

        new_arr = np.zeros((len(trials), self._N, self._N))
        for t, trial in enumerate(trials):
            new_arr[t] = np.cov(sub_spikes[trial])
        new_arr /= new_arr.max()

        if save_png_name is not None:
            new_arr = new_arr.reshape(len(trials) * self._N, self._N)
            save_matrix_whole_canvas(new_arr, save_png_name + '.png', cmap='gray')
        else:
            return new_arr

    # i/o

    def save(self, file_name='spikes', extra=None, overwrite=False):
        """
        Saves contents to file.

        Parameters
        ----------
        file_name : str, optional
            File name to save to (default 'spikes')
        extra : dict, optional
            Extra information to save to file (default None)

        Returns
        -------
        Nothing
        """
        return super(Spikes, self)._save(file_name=file_name,
                                         attributes=self._SAVE_ATTRIBUTES_V1, version=self._SAVE_VERSION,
                                         extra=extra)

    @classmethod
    def load(cls, file_name='spikes', load_extra=False):
        """
        Loads contents from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            spikes = Spikes.load('file_name')

        Parameters
        ----------
        file_name : str, optional
            File name to load from (default 'spikes')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        spikes : :class:`.Spikes`
            Instance of :class:`.Spikes` if loaded, `None` upon error
        """
        return super(Spikes, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        # internal function to load v1 file format
        hdlog.debug('Loading Spikes, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<Spikes: {n} neurons, {m} bins, {t} trials>'.format(n=self._N, m=self._M, t=self._T)


# end of source
