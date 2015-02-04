# -*- coding: utf-8 -*-
"""
    hdnet.patterns
    ~~~~~~~~~~~~~~

    Record / counts of fixed-points of Hopfield network.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np
from util import hdlog, Restoreable


class Counter(Restoreable, object):
    """ Catalogues binary vectors and their prevalence

    Parameters
        counts:  dictionary of (key_for_pattern, value) = (string, object)
        patterns:  list of binary vectors in order of discovery
        lookup_patterns:  dictionary of (key_for_pattern, patterns index)
        sequence:  list over trials T of (lists of indices where fp found in spikes dataset)
    """

    _SAVE_ATTRIBUTES_V1 = ['_counts', '_patterns', '_lookup_patterns',
                        '_sequence', '_skipped_patterns', '_seen_sequence']
    _SAVE_VERSION = 1

    @staticmethod
    def key_for_pattern(pattern):
        """ Transforms a numpy binary array into a string """
        # TODO: document
        return ''.join(str(k) for k in pattern.astype(np.int).ravel())

    @staticmethod
    def pattern_for_key(key):
        """ Transforms string into a numpy binary array """
        # TODO: document
        return np.array([int(k) for k in list(key)])

    def __init__(self, counter=None, save_sequence=True):
        object.__init__(self)
        Restoreable.__init__(self)
        self._counts = {}
        self._patterns = []
        self._lookup_patterns = {}
        self._sequence = []
        self._save_sequence = save_sequence
        self._skipped_patterns = 0
        self._seen_sequence = []

        if counter is not None:
            self.merge_counts(counter)

    @property
    def counts(self):
        # TODO: document
        return self._counts

    @property
    def patterns(self):
        # TODO: document
        return self._patterns

    @property
    def lookup_patterns(self):
        # TODO: document
        return self._lookup_patterns

    @property
    def sequence(self):
        # TODO: document
        return self._sequence

    @property
    def skipped_patterns(self):
        # TODO: document
        return self._skipped_patterns

    @property
    def seen_sequence(self):
        # TODO: document
        return self._seen_sequence

    def __add__(self, acc):
        # TODO: document
        return self.merge_counts(acc)

    def __len__(self):
        # TODO: document
        return len(self._counts.keys())

    def merge_counts(self, counter):
        """ Combine counts with another Counter """
        # TODO: document
        for key in counter.counts.keys():
            key_ = Counter.key_for_pattern(Counter.pattern_for_key(key))
            self.add_key(key_, counter.counts[key])
        return self

    def add_key(self, key, value=1):
        # TODO: document
        if key in self._counts:
            self._counts[key] += value
            return True
        self._counts[key] = value
        self._lookup_patterns[key] = len(self._patterns)
        self._patterns.append(key)
        return False

    def chomp(self, X, add_new=True, rotate=None):
        """ M x N numpy array X as input (N neurons, M vects) """
        # TODO: document
        for x in X:
            self.chomp_vector(x, add_new=add_new, rotate=rotate)

    def chomp_spikes(self, spikes, add_new=True, window_size=1, trials=None, rotate=None):
        # TODO: document
        if rotate and add_new:
            self._skipped_patterns = 0
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_vector(self, x, add_new=True, rotate=None):
        """ stores bin vects (originally x) y and order of occurence """
        # TODO: document
        bin_x = Counter.key_for_pattern(x)

        numrot = 0
        if rotate:
            xrot = x.reshape(rotate)
            found = bin_x in self._counts
            while not found and numrot < rotate[1]:
                xrot = np.roll(xrot, 1, axis=1)
                bin_x = Counter.key_for_pattern(xrot.reshape(x.shape))
                found = bin_x in self._counts
                numrot += 1

            if not found:
                bin_x = Counter.key_for_pattern(x)
            elif numrot > 0:
                self._skipped_patterns += 1

        if bin_x in self._counts:
            self._seen_sequence.append(1)
            self._counts[bin_x] += 1
            if self._save_sequence:
                self._sequence.append(self._lookup_patterns[bin_x])
            new_pattern = False
        else:
            self._seen_sequence.append(0)
            if add_new:
                self._patterns.append(bin_x)
                self._lookup_patterns[bin_x] = len(self._patterns) - 1
                self._counts[bin_x] = 1
                if self._save_sequence:
                    self._sequence.append(self._lookup_patterns[bin_x])
            new_pattern = True
        return bin_x, new_pattern, numrot

    def to_prob_vect(self, parent=None):
        """ if parent (= counter object) present then return prob vector in that space """
        # TODO: document
        if parent is not None:
            values = np.zeros(len(parent.counts))
            for i in xrange(len(self._counts)):
                if self._counts.keys()[i] in parent.counts:
                    values[parent.lookup_patterns[self._counts.keys()[i]]] = self._counts.values()[i]
        else:
            values = np.array(self._counts.values())

        probs = 1. * values / values.sum()
        return probs

    def entropy(self):
        # TODO: document
        probs = self.to_prob_vect()
        return -(probs * np.log2(probs)).sum()

    def pattern_to_binary_matrix(self, m):
        # TODO: document
        key = self._patterns[m]
        return Counter.pattern_for_key(key)

    def top_binary_matrices(self, m):
        """ finds top m likely memories """
        # TODO: document
        top_binary = []
        idx = np.array(self._counts.values()).argsort()[-m:]
        for i in idx:
            top_binary.append(self.pattern_to_binary_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return top_binary

    def mem_triggered_stim_avgs(self, stimulus):
        """ returns the average stimulus appearing when a given binary pattern appears """
        # TODO: document
        stim_avgs = []
        stm_arr = stimulus.stimulus_arr
        arr = np.zeros((stm_arr.shape[0] * stm_arr.shape[1],) + stm_arr.shape[2:])

        for t in xrange(stm_arr.shape[0]):
            arr[t * stm_arr.shape[1]:(t + 1) * stm_arr.shape[1]] = stm_arr[t]

        for c, pattern in enumerate(self._patterns):
            idx = (self._sequence == c)
            stim_avgs.append(arr[idx].mean(axis=0))
            # if c > 1:
            #     stop

        return stim_avgs

    # representation

    def __repr__(self):
        return '<Counter: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))

    # i/o

    def save(self, filename='counter', extra=None):
        """ save as numpy array .npz file """
        # TODO: document
        return super(Counter, self).save(filename=filename,
                                         attributes=self._SAVE_ATTRIBUTES_V1, version=self._SAVE_VERSION,
                                         extra=extra)

    @classmethod
    def load(cls, filename='counter', load_extra=False):
        # TODO: document
        return super(Counter, cls).load(filename=filename, load_extra=load_extra)

    def load_v1(self, contents, load_extra=False):
        hdlog.debug('loading Counter patterns, format version 1')
        return Restoreable.load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    @classmethod
    def load_legacy(cls, filename='patterns_raw'):
        hdlog.info("loading Counter patterns from legacy file '%s'" % filename)
        instance = cls()
        contents = np.load(filename)
        instance._counts = dict(zip(contents['count_keys'], contents['count_values']))
        instance._patterns = contents['fp_list']
        instance._lookup_patterns = dict(zip(contents['lookup_fp_keys'], contents['lookup_fp_values']))
        instance._sequence = contents['sequence']
        contents.close()
        return instance


class PatternsRaw(Counter):
    # TODO: document

    def __init__(self, patterns_raw=None, save_sequence=True):
        super(PatternsRaw, self).__init__(counter=patterns_raw, save_sequence=save_sequence)

    # representation

    def __repr__(self):
        return '<PatternsRaw: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))

    # i/o

    def save(self, filename='patterns_raw', extra=None):
        # TODO: document
        super(PatternsRaw, self).save(filename, extra=extra)

    @classmethod
    def load(cls, filename='patterns_raw', load_extra=False):
        # TODO: document
        instance, extra = super(PatternsRaw, cls).load(filename=filename, load_extra=True)
        if load_extra:
            return instance, extra
        else:
            return instance


class PatternsHopfield(Counter):
    """ record / counts of fixed-points of Hopfield network 
    
        fixed_points / memories are stored in dictionary self.counts

    Parameters
        learner: hopfield network and learning params
        mtas: dict taking bin vects v to sums of orignal binary vectors converging to v
    """
    _SAVE_ATTRIBUTES_V1 = ['_counts', '_patterns', '_lookup_patterns',
                        '_sequence', '_skipped_patterns', '_seen_sequence',
                        '_mtas', '_mtas_raw']
    _SAVE_VERSION = 1

    def __init__(self, learner=None, patterns_hopfield=None, save_sequence=True, save_raw=True):
        super(PatternsHopfield, self).__init__(save_sequence=save_sequence)

        self._learner = learner or None
        self._mtas = {}
        self._mtas_raw = {}
        self._save_raw = save_raw

        if patterns_hopfield is not None:
            self.merge_counts(patterns_hopfield)

    @property
    def mtas(self):
        # TODO: document
        return self._mtas

    @property
    def mtas_raw(self):
        # TODO: document
        return self._mtas_raw

    def add_key(self, key, value=1, raw=None):
        # TODO: document
        known_key = super(PatternsHopfield, self).add_key(key, value)

        if known_key:
            self._mtas[key] += raw
            if self._save_raw:
                self._mtas_raw[key].append(raw)
        else:
            self._mtas[key] = raw
            if self._save_raw:
                self._mtas_raw[key] = [raw]

        return known_key

    def merge_counts(self, patterns_hopfield):
        """ combine your counts with another counter """
        # TODO: document
        for key in patterns_hopfield.counts.keys():
            o_key = Counter.key_for_pattern(self._learner.network(Counter.pattern_for_key(key)))
            # TODO fix merging wrt mtas / raw
            if o_key in patterns_hopfield.mtas:
                raw = patterns_hopfield.mtas[o_key]
            else:
                raw = None
            self.add_key(o_key, patterns_hopfield.counts[key], raw)
        return self

    def chomp(self, X, add_new=True, rotate=None):
        """ M x N numpy array X as input (N neurons, M vects) """
        # TODO: document
        # TODO warn if no network
        Y = self._learner.network(X)
        for x, y in zip(X, Y):
            self.chomp_vector(x, y, add_new=add_new, rotate=rotate)

    def chomp_vector(self, x, y, add_new=True, rotate=None):
        """ stores bin vects (originally x) y and order of occurence """
        # TODO: document
        bin_y, new_pattern, numrot = super(PatternsHopfield, self).chomp_vector(y, add_new=add_new, rotate=rotate)

        if rotate and numrot > 0:
            xrot = x.reshape(rotate)
            xrot = np.roll(xrot, numrot, axis=1)
            x = Counter.key_for_pattern(xrot.reshape(x.shape))

        if new_pattern:
            self._mtas[bin_y] = x
            self._mtas_raw[bin_y] = [x]
        elif add_new:
            self._mtas[bin_y] += x
            self._mtas_raw[bin_y].append(x)

        return x, bin_y, new_pattern, numrot

    def apply_dynamics(self, spikes, add_new=True, window_size=1, trials=None, reshape=True):
        # TODO: document
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        # TODO warn if no network
        Y = self._learner.network(X)
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

    def pattern_to_mta_matrix(self, m):
        # TODO: document
        key = self._patterns[m]
        return self._mtas[key] / self._counts[key]

    def pattern_to_raw_patterns(self, m):
        # TODO: document
        key = self._patterns[m]
        return np.array(self._mtas_raw[key])

    def pattern_to_mta_std(self, m):
        # TODO: document
        raw_patterns = self.pattern_to_raw_patterns(m)
        return raw_patterns.std(axis=0)

    def top_mta_matrices(self, count):
        """ finds (top count likely memory)-triggered averages """
        # TODO: document
        top_mtas = []
        idx = np.array(self._counts.values()).argsort()[-count:]
        for i in idx:
            top_mtas.append(self.pattern_to_mta_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return np.array(top_mtas)

    def pattern_to_trial_raster(self, m, start=0, stop=None, trials=None):
        # TODO: document
        stop = stop or self._learner.spikes.M
        trials = trials or range(self._learner.spikes.T)
        key = self._patterns[m]
        sequence = np.array(self._sequence).reshape(self._learner.spikes.T,
                                                   self._learner.spikes.M - self._learner.window_size + 1)[:, start:stop]

        hits = (sequence == m).astype(int)
        return hits

    def approx_basin_size(self, max_corrupt_bits=1):
        """ average bits corruption memory can stand """
        # TODO: document
        # TODO implement
        pass

    # representation

    def __repr__(self):
        return '<PatternsHopfield: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))

    # i/o

    def save(self, filename='patterns_hopfield', extra=None):
        # TODO: document
        super(PatternsHopfield, self).save(filename=filename, extra=extra)

    @classmethod
    def load(cls, filename='patterns_hopfield', load_extra=False):
        # TODO: document
        return super(PatternsHopfield, cls).load(filename=filename, load_extra=load_extra)

    def load_v1(self, contents, load_extra=False):
        hdlog.debug('loading PatternsHopfield patterns, format version 1')
        return Restoreable.load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    @classmethod
    def load_legacy(cls, filename='patterns_hopfield'):
        hdlog.info("loading PatternHopfield patterns from legacy file '%s'" % filename)
        instance = cls()
        contents = np.load(filename)
        instance._counts = dict(zip(contents['count_keys'], contents['count_values']))
        instance._patterns = contents['fp_list']
        instance._lookup_patterns = dict(zip(contents['lookup_fp_keys'], contents['lookup_fp_values']))
        instance._sequence = contents['sequence']
        instance._mtas = dict(zip(contents['stas_keys'], contents['stas_values']))
        instance._sequence = contents['sequence']
        contents.close()
        return instance


# end of source
