# -*- coding: utf-8 -*-
"""
    hdnet.patterns
    ~~~~~~~~~~~~~~

    Record / counts of fixed-points of Hopfield network.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np

from counter import Counter


class Patterns(Counter):
    """ record / counts of fixed-points of Hopfield network 
    
        fixed_points / memories are stored in dictionary self.counts

    Parameters
        learner: hopfield network and learning params
        stas: dict taking bin vects v to sums of orignal binary vectors converging to v
    """

    def __init__(self, learner=None, counter=None, save_fp_sequence=False):
        self.learner = learner or None
        
        super(Patterns, self).__init__(save_fp_sequence=save_fp_sequence)
        
        self.stas = {}
        
        if counter is not None:
            self.merge_counts(counter)


    def merge_counts(self, counter):
        """ combine your counts with another counter """
        for key in counter.counts.keys():
            key_ = self.key(self.chomp_dynamics(self.reverse_key(key)))
            if hasattr(counter, 'stas'):
                sta = counter.stas[key_]
            else:
                sta = None
            self.add_key(key_, counter.counts[key], sta)
        return self

    def add_key(self, key, value=1, sta=None):
        known_key = super(Patterns, self).add_key(key, value)

        if sta is None:
            return known_key

        if known_key:
            self.stas[key] += sta
        else:
            self.stas[key] = sta

        return known_key

    def chomp_dynamics(self, X):
        """ X is M x N, converge dynamics before counting binary patterns """
        return self.learner.network(X)


    def chomp(self, X, add_new=True, rotate=None):
        """ M x N numpy array X as input (N neurons, M vects) """
        Y = self.chomp_dynamics(X)
        for x, y in zip(X, Y):
            self.chomp_vector(x, y, add_new=add_new, rotate=rotate)

    def chomp_vector(self, x, y, add_new=True, rotate=None):
        """ stores bin vects (originally x) y and order of occurence """
        bin_y, new_pattern, numrot = super(Patterns, self).chomp_vector(y, add_new=add_new, rotate=rotate)

        if rotate and numrot>0:
            xrot=x.reshape(rotate)
            xrot=np.roll(xrot, numrot, axis=1)
            x=self.key(xrot.reshape(x.shape))

        if not new_pattern:
            self.stas[bin_y] += x
        elif add_new:
            self.stas[bin_y] = x

        return x, bin_y, new_pattern, numrot


    def apply_dynamics(self, spikes, add_new=True, window_size=1, trials=None, reshape=True):
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        Y = self.chomp_dynamics(X)
        if reshape:
            N = Y.shape[1]
            if trials is None:
                T = spikes.T
            else:
                T = len(trials)
            M = Y.shape[0] / T
            Y_ = np.zeros((T, N, M))
            for n in xrange(N):
                Y_[:, n, :] = Y[:, n].reshape((T, M))
            Y = Y_
        return Y

    def fp_to_trial_raster(self, m, start=0, stop=None, trials=None):
        stop = stop or self.learner.spikes.M
        trials = trials or range(self.learner.spikes.T)
        key = self.fp_list[m]
        sequence = np.array(self.sequence).reshape(self.learner.spikes.T, 
            self.learner.spikes.M - self.learner.window_size + 1)[:, start:stop]

        hits = (sequence == m).astype(int)

        return hits

    def approx_basin_size(self, max_corrupt_bits=1):
        """ average bits corruption memory can stand """
        pass

    def save(self, filename='counter'):
        """ save as numpy array .npz file
         TODO: add saving of STAS in Patterns """
        np.savez(filename, count_keys=self.counts.keys(), count_values=self.counts.values(),
            fp_list=self.fp_list, lookup_fp_keys=self.lookup_fp.keys(),
            lookup_fp_values=self.lookup_fp.values(), stas_keys=self.stas.keys(),
            stas_values=self.stas.values(), sequence=self.sequence, skippedpatterns=self.skippedpatterns)


    def load(self, filename='counter'):
        filename += '.npz'
        arr = np.load(filename)
        self.counts = dict(zip(arr['count_keys'], arr['count_values']))
        self.fp_list = arr['fp_list']
        self.lookup_fp = dict(zip(arr['lookup_fp_keys'], arr['lookup_fp_values']))
        self.sequence = arr['sequence']
        self.stas = dict(zip(arr['stas_keys'], arr['stas_values']))
