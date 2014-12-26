# -*- coding: utf-8 -*-
"""
    hdnet.spikes
    ~~~~~~~~~~~~

    Spikes class handling multi-neuron , multi-trial spike trains.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

__all__ = ('Spikes')

import os
import numpy as np


class Spikes(object):
    """ Class for handling binary time-series datasets

    Parameters
        npz_file: filename of N x M numpy array containing M time bin steps of N neurons' spikes
                        OR T x N x M numpy array of T trials, repeated stimulus
        spikes_arr:  T x N x M array of spikes (T always present even when T = 1)

        preprocess: makes data into binary {0,1} (Heaviside)
                    override for other operations on raw data
    """

    def __init__(self, spikes_arr=None, npz_file=None, mat_file=None, spk_files=None, spk_folder=None, bin_size=1,
                 preprocess=True):
        self.filename = npz_file or ''

        self.T = 0
        self.N = 0
        self.M = 0

        # TODO: instead of different parameters for different file formats use just two parameters, file and file_format

        if spikes_arr is not None:
            self.spikes_arr = spikes_arr
        elif npz_file is not None:
            if os.path.isfile(npz_file):
                print('File found. Loading %s') % npz_file
                self.filename = npz_file
                tmp = np.load(npz_file)
                self.spikes_arr = tmp[tmp.keys()[0]]
            else:
                print('Not a file.')
                return
        elif mat_file is not None:
            import scipy.io

            mat = scipy.io.loadmat(mat_file)
            self.spikes_arr = mat[mat.keys()[0]]
        elif spk_files is not None:
            from bitstring import Bits

            self.neuron_to_file = []
            time_stamps = []
            self.bin_size = bin_size or 1

            for fname in spk_files:
                self.neuron_to_file.append(fname)
                f = open(fname, 'rb')
                p = Bits(f)
                fmt = str(p.length / 64) + ' * (intle:64)'
                time_stamps.append(p.unpack(fmt))
            self.load_from_spikes_times(time_stamps, bin_size=self.bin_size)
        elif spk_folder is not None:
            from bitstring import Bits

            self.neuron_to_file = []
            time_stamps = []
            self.bin_size = bin_size or 1
            fns = os.listdir(spk_folder)

            for i, fn in enumerate(fns):
                ext = os.path.splitext(fn)[1]
                if ext in ('.spk', ):  #  Blanche spike format
                    self.neuron_to_file.append(fn)
                    f = open(os.path.join(spk_folder, fn), 'rb')
                    p = Bits(f)
                    fmt = str(p.length / 64) + ' * (intle:64)'
                    time_stamps.append(p.unpack(fmt))
                self.load_from_spikes_times(time_stamps, bin_size=self.bin_size)
        else:
            return

        spikes_shape = self.spikes_arr.shape
        if len(spikes_shape) == 2:  # single trial
            self.spikes_arr = self.spikes_arr.reshape((1, spikes_shape[0], spikes_shape[1]))

        self.T = self.spikes_arr.shape[0]
        self.N = self.spikes_arr.shape[1]
        self.M = self.spikes_arr.shape[2]

        if preprocess: self.preprocess()

    def restrict_to_most_active_neurons(self, top_neurons=None):
        """ (does not make a copy) if top_neurons is None: sorts the spike_arr """
        self.N = top_neurons or self.N
        activity = self.spikes_arr[:, :, :].mean(axis=0).mean(axis=1)
        idx = activity.argsort()
        self.idx = idx[-self.N:]
        self.mean_activities = activity[self.idx]
        self.spikes_arr = self.spikes_arr[:, idx[-self.N:], :]
        return self

    def to_windowed(self, window_size=1, trials=None, reshape=False):
        """ returns new Spikes object of 3d numpy arr of windowed spike trains:
                X:   T (num trials) x (window_size * N) x  (M - window_size + 1)
                                        binary vector out of a spike time series
            reshape: returns T(M - window_size + 1) x (ws * N) numpy binary vector
        """
        trials = trials or range(self.T)
        X = np.zeros((len(trials), window_size * self.N, self.M - window_size + 1))
        for c, t in enumerate(trials):
            for i in xrange(0, self.M - window_size + 1):
                X[c, :, i] = self.spikes_arr[t, :, i:window_size + i].ravel()

        if reshape:
            Y = np.zeros((X.shape[0] * X.shape[2], X.shape[1]))
            tot = 0
            for t in xrange(len(trials)):
                for j in xrange(X.shape[2]):
                    Y[tot, :] = X[t, :, j]
                    tot += 1
            return Y
        return Spikes(spikes_arr=X)

    def load_from_spikes_times(self, spike_times_lists, bin_size=1):
        """ loads a spike train from a list of arrays of spike times
            bin_size: in milisec
                - the jth item in the list corresponds to the jth neuron
                  it is the 1d array of spike times (micro sec) for that neuron
        """
        if len(spike_times_lists) == 0: return
        self.max_milisec = - np.inf
        for spike_times in spike_times_lists:
            milisec = 1. * (spike_times[-1]) / (10 ** 3)
            self.max_milisec = max(self.max_milisec, milisec)
        self.spikes_arr = np.zeros((len(spike_times_lists), np.int(self.max_milisec) / bin_size + 1))
        for c, spike_times in enumerate(spike_times_lists):
            for spike_time in spike_times:
                a = int(spike_time / (1000. * bin_size))
                if a < self.spikes_arr.shape[1]:
                    self.spikes_arr[c, a] = 1

    def preprocess(self):
        """ converts to binary """
        self.spikes_arr = np.double((np.sign(self.spikes_arr) + 1) // 2)

    def rasterize(self, trials=None, start=0, stop=None, save_png_name=None):
        """ return *new* (copied) numpy array of size (TN x M)
            trials: e.g. [1, 5, 6], None is all
            save_png_name: if not None then only saves (PIL needs to be installed)
        """
        stop = stop or self.M
        trials = trials or range(self.T)
        sub_spikes_arr = self.spikes_arr[trials, :, start:stop]

        if save_png_name is not None:
            from PIL import Image

            new_arr = np.round(sub_spikes_arr * 255.).reshape((len(trials) * self.N, stop - start))
            new_arr = np.insert(new_arr, range(self.N, len(trials) * self.N, self.N), 127., axis=0)
            im_png = Image.fromarray(new_arr).convert('L')
            im_png.save(save_png_name + '.png')
        else:
            return sub_spikes_arr.copy().reshape((len(trials) * self.N, stop - start))

    def covariance(self, trials=None, start=0, stop=None, save_png_name=None):
        """ return *new* numpy array of size (T x N x N) which is covariance matrix betwn neurons
            trials: e.g. [0, 1, 5, 6], None is all
            save_png_name: if not None then only saves (PIL needs to be installed)
        """
        stop = stop or self.M
        trials = trials or range(self.T)
        sub_spikes_arr = self.spikes_arr[trials, :, start:stop]

        new_arr = np.zeros((len(trials), self.N, self.N))
        for t, trial in enumerate(trials):
            new_arr[t] = np.cov(sub_spikes_arr[trial])

        if save_png_name is not None:
            from PIL import Image
            new_arr = new_arr.reshape(len(trials) * self.N, self.N)
            new_arr /= new_arr.max()
            im_png = Image.fromarray(255. * new_arr).convert('L')
            im_png.save(save_png_name + '.png')
        else:
            return new_arr
