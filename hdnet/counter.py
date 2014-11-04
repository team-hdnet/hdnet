# -*- coding: utf-8 -*-
"""
    hdnet.counter
    ~~~~~~~~~~~~~

    Counter class for counting patterns in spike trains

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np

__all__ = ('Counter')

class Counter(object):
    """ Catalogues binary vectors and their prevalence

    Parameters
        counts:  dictionary of (key, value) = (string, object)
        fp_list:  list of binary vectors in order of discovery
        lookup_fp:  dictionary of (key, fp_list index)
        sequence:  list over trials T of (lists of indices where fp found in spikes dataset)
    """

    def __init__(self, save_fp_sequence=False):
        self.counts = {}
        self.fp_list = []
        self.lookup_fp = {}
        self.sequence = []
        self.save_fp_sequence = save_fp_sequence
        self.skippedpatterns = 0
        self.numpatterns = 0
        self.seensequence=[]

    def __add__(self, acc):
        return self.merge_counts(acc)

    def __len__(self):
        return len(self.counts.keys())

    def add_key(self, key, value=1):
        if self.counts.has_key(key):
            self.counts[key] += value
            return True
        self.counts[key] = value
        self.lookup_fp[key] = len(self.fp_list)
        self.fp_list.append(key)
        return False

    def key(self, bin):
        """ takes numpy binary array bin and makes a string """
        return "".join(str(k) for k in bin.astype(np.int).ravel())

    def reverse_key(self, key):
        return np.array([int(k) for k in list(key)])

    def chomp_spikes(self, spikes, add_new=True, window_size=1, trials=None, rotate=None):
        if rotate and add_new:
            self.skippedpatterns = 0
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_patterns(self, X, add_new=True, window_size=1, trials=None, rotate=None):
        if rotate and add_new:
            self.skippedpatterns = 0
        #X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_spikes_bernoulli(self, spikes, add_new=True, window_size=1, bin_size=1, trials=None, rotate=None):
        if rotate and add_new:
            self.skippedpatterns = 0
        X = spikes.to_windowed_bernoulli(window_size=window_size, bin_size=bin_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_vector(self, x, add_new=True, rotate=None):
        """ stores bin vects (originally x) y and order of occurence """
        bin_x = self.key(x)

        numrot=0
        if rotate:
            xrot=x.reshape(rotate)
            found=self.counts.has_key(bin_x)
            while not found and numrot<rotate[1]:
                xrot=np.roll(xrot, 1, axis=1)
                bin_x=self.key(xrot.reshape(x.shape))
                found=self.counts.has_key(bin_x)
                numrot+=1

            if not found:
                bin_x = self.key(x)
            elif numrot>0:
                self.skippedpatterns+=1

        if self.counts.has_key(bin_x):
            self.seensequence.append(1)
            self.counts[bin_x] += 1
            if self.save_fp_sequence:
                self.sequence.append(self.lookup_fp[bin_x])
            new_pattern = False
        else:
            self.seensequence.append(0)
            if add_new:
                self.fp_list.append(bin_x)
                self.lookup_fp[bin_x] = len(self.fp_list) - 1
                self.counts[bin_x] = 1
                if self.save_fp_sequence:
                    self.sequence.append(self.lookup_fp[bin_x])
            new_pattern = True
        return bin_x, new_pattern, numrot

    def chomp(self, X, add_new=True, rotate=None):
        """ M x N numpy array X as input (N neurons, M vects) """
        for x in X:
            self.chomp_vector(x, add_new=add_new, rotate=rotate)

    def merge_counts(self, counter):
        """ combine your counts with another counter """
        for key in counter.counts.keys():
            key_ = self.key(self.reverse_key(key))
            self.add_key(key_, counter.counts[key])
        return self

    def entropy(self):
        values = np.array(self.counts.values())
        probs = 1. * values / values.sum()
        return -(probs * np.log2(probs)).sum()

    def to_prob_vect(self, parent=None):
        """ if parent (= counter object) present then return prob vector in that space """
        if parent is not None:
            values = np.zeros(len(parent.counts))
            for i in xrange(len(self.counts)):
                if parent.counts.has_key(self.counts.keys()[i]):
                    values[parent.lookup_fp[self.counts.keys()[i]]] = self.counts.values()[i]
        else:
            values = np.array([self.counts[self.fp_list[i]] for i in xrange(len(self.fp_list))])

        probs = 1. * values / values.sum()
        return probs

    def save(self, filename='counter'):
        """ save as numpy array .npz file 
         TODO: add saving of STAS in Patterns """
        np.savez(filename, count_keys=self.counts.keys(), count_values=self.counts.values(),
            fp_list=self.fp_list, lookup_fp_keys=self.lookup_fp.keys(),
            lookup_fp_values=self.lookup_fp.values(), sequence=self.sequence, skippedpatterns=self.skippedpatterns)

    def load(self, filename='counter'):
        filename += '.npz'
        arr = np.load(filename)
        self.counts = dict(zip(arr['count_keys'], arr['count_values']))
        self.fp_list = arr['fp_list']
        self.lookup_fp = dict(zip(arr['lookup_fp_keys'], arr['lookup_fp_values']))
        self.sequence = arr['sequence']

#######################################
# TODO: incorporate this STA stuff better
#######################################

    def fp_to_binary_matrix(self, m):
        key = self.fp_list[m]
        return self.reverse_key(key).reshape(self.learner.spikes.N, self.learner.window_size)

    def fp_to_sta_matrix(self, m):
        key = self.fp_list[m]
        return self.stas[key].reshape(self.learner.spikes.N, self.learner.window_size) / self.counts[key]

    def top_binary_matrices(self, m):
        """ finds top m likely memories """
        top_binary = []
        idx = np.array(self.counts.values()).argsort()[-m:]
        for i in idx:
            top_binary.append(self.fp_to_binary_matrix(self.lookup_fp[self.counts.keys()[i]]))
        return top_binary

    def top_sta_matrices(self, m):
        """ finds (top m likely memory)-triggered averages """
        top_stas = []
        idx = np.array(self.counts.values()).argsort()[-m:]
        for i in idx:
            top_stas.append(self.fp_to_sta_matrix(self.lookup_fp[self.counts.keys()[i]]))
        return top_stas



