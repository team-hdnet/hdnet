# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.patterns
    ~~~~~~~~~~~~~~

    Record / counts of fixed-points of Hopfield network.

"""

import os
import numpy as np
from hdnet.spikes import Spikes
from util import hdlog, Restoreable


class Counter(Restoreable, object):
    """
    Catalogues binary vectors and their prevalence.

    Parameters
    ----------
    counter : :class:`.Counter`, optional
        Counter object to merge with (default None)
    save_sequence : bool, optional
        Flag whether to save the sequence of pattern
        labels, labels given by order of appearance (default True)

    Returns
    -------
    counter : :class:`.Counter`.
        Instance of class :class:`.Counter`
    """

    _SAVE_ATTRIBUTES_V1 = ['_counts', '_patterns', '_lookup_patterns',
                        '_sequence', '_skipped_patterns', '_seen_sequence']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Counter'

    @staticmethod
    def key_for_pattern(pattern):
        """
        Computes key (as string) of binary pattern `pattern`.
        Reverse loopup for method :meth:`pattern_for_key`.

        Returns
        -------
        key : str
            String representation of binary pattern
        """
        return ''.join(str(k) for k in pattern.astype(np.int).ravel())

    @staticmethod
    def pattern_for_key(key):
        """
        Computes binary pattern (as numpy matrix) from string
        representation `key`.
        Reverse loopup for method :meth:`key_for_pattern`.

        Returns
        -------
        pattern : numpy array
            binary pattern (as numpy matrix)
        """
        return np.array([int(k) for k in list(key)])

    @staticmethod
    def pattern_distance_jaccard(a, b):
        """
        Computes a distance measure for two binary patterns based on their
        Jaccard-Needham distance, defined as

        .. math::

            d_J(a,b) = 1 - J(a,b) = \\frac{|a \\cup b| - |a \\cap b|}{|a \\cup b|}.

        The similarity measure takes values on the closed interval [0, 1],
        where a value of 1 is attained for disjoint, i.e. maximally dissimilar
        patterns a and b and a value of 0 for the case of :math:`a=b`.

        Parameters
        ----------
        a : list or array, int or bool
            Input pattern
        b : list or array, int or bool
            Input pattern

        Returns
        -------
        dist : double
            Jaccard distance between `a` and `b`.
        """
        # Note: code taken from scipy. Duplicated as only numpy references wanted for base functionality
        a = np.atleast_1d(a).astype(bool)
        b = np.atleast_1d(b).astype(bool)
        dist = (np.double(np.bitwise_and((a != b), np.bitwise_or(a != 0, b != 0)).sum())
                / np.double(np.bitwise_or(a != 0, b != 0).sum()))
        return dist

    @staticmethod
    def pattern_distance_hamming(a, b):
        """
        Computes a distance measure for two binary patterns based on their
        normed Hamming distance, defined as

        .. math::

            d_H(a,b)=\\frac{1}{n}|\\left\\{j \\in \\{1,\\dots,n\\}\\mid a_j \\neq b_j \\right\\}|,

        if both :math:`a` and :math:`b` have length :math:`n`.

        The similarity measure takes values on the closed interval [0, 1],
        where a value of 0 is attained for disjoint, i.e. maximally dissimilar
        patterns a and b and a value of 1 for the case of :math:`a=b`.

        Parameters
        ----------
        a : list or array, int or bool
            Input pattern
        b : list or array, int or bool
            Input pattern

        Returns
        -------
        dist : double
            Normed Hamming distance between `a` and `b`.
        """
        # Note: code taken from scipy. Duplicated as only numpy references wanted for base functionality
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        return (a != b).mean()

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
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._counts

    @property
    def patterns(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._patterns

    @property
    def lookup_patterns(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._lookup_patterns

    @property
    def sequence(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._sequence

    @property
    def skipped_patterns(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._skipped_patterns

    @property
    def seen_sequence(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self._seen_sequence

    def __add__(self, acc):
        """
        Missing documentation
        
        Parameters
        ----------
        acc : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return self.merge_counts(acc)

    def __len__(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        return len(self._counts.keys())

    def merge_counts(self, counter):
        """
        Combine counts with another Counter
        
        Parameters
        ----------
        counter : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO: document
        for key in counter.counts.keys():
            key_ = Counter.key_for_pattern(Counter.pattern_for_key(key))
            self.add_key(key_, counter.counts[key])
        return self

    def add_key(self, key, value=1):
        """
        Missing documentation
        
        Parameters
        ----------
        key : Type
            Description
        value : int, optional
            Description (default 1)
        
        Returns
        -------
        Value : Type
            Description
        """
        if key in self._counts:
            self._counts[key] += value
            return True
        self._counts[key] = value
        self._lookup_patterns[key] = len(self._patterns)
        self._patterns.append(key)
        return False

    def chomp(self, X, add_new=True, rotate=None):
        """
        M x N numpy array X as input (N neurons, M vects)

        Parameters
        ----------
        X : Type
            Description
        add_new : bool, optional
            Description (default True)
        rotate : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        for x in X:
            self.chomp_vector(x, add_new=add_new, rotate=rotate)

    def chomp_spikes(self, spikes, add_new=True, window_size=1, trials=None, rotate=None):
        """
        Missing documentation
        
        Parameters
        ----------
        spikes : Type
            Description
        add_new : bool, optional
            Description (default True)
        window_size : int, optional
            Description (default 1)
        trials : Type, optional
            Description (default None)
        rotate : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        if rotate and add_new:
            self._skipped_patterns = 0
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_vector(self, x, add_new=True, rotate=None):
        """
        stores bin vects (originally x) y and order of occurence

        Parameters
        ----------
        x : Type
            Description
        add_new : bool, optional
            Description (default True)
        rotate : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
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


    def pattern_to_binary_matrix(self, m):
        """
        Returns representation of pattern as binary vector.
        
        Parameters
        ----------
        m : str
            Key of pattern
        
        Returns
        -------
        pattern : numpy array
            Representation of pattern as binary vector
        """
        key = self._patterns[m]
        return Counter.pattern_for_key(key)

    def top_binary_matrices(self, m):
        """
        Returns top m likely patterns.
        
        Parameters
        ----------
        m : int
            Number of top likely patterns to return
        
        Returns
        -------
        patterns : numpy array
            m top likely patterns
        """
        top_binary = []
        idx = np.array(self._counts.values()).argsort()[-m:]
        for i in idx:
            top_binary.append(self.pattern_to_binary_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return top_binary

    def mem_triggered_stim_avgs(self, stimulus):
        """
        Returns the average stimulus appearing when a given binary pattern appears.
        
        Parameters
        ----------
        stimulus : :class:`.Stimulus`
            Instance of :class:`.Stimulus` class to query
        
        Returns
        -------
        averages : numpy array
            Stimulus average calculated
        """
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

    # i/o

    def save(self, file_name='counter', extra=None):
        """
        Saves contents to file.

        Parameters
        ----------
        file_name : str, optional
            File name to save to (default 'counter')
        extra : dict, optional
            Extra information to save to file (default None)

        Returns
        -------
        Nothing
        """
        return super(Counter, self)._save(file_name=file_name,
                                         attributes=self._SAVE_ATTRIBUTES_V1, version=self._SAVE_VERSION,
                                         extra=extra)

    @classmethod
    def load(cls, file_name='counter', load_extra=False):
        """
        Loads contents from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            counter = Counter.load('file_name')

        Parameters
        ----------
        file_name : str, optional
            File name to load from (default 'counter')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        counter : :class:`.Counter`
            Instance of :class:`.Counter` if loaded, `None` upon error
        """
        return super(Counter, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('Loading Counter patterns, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    @classmethod
    def load_legacy(cls, file_name='counter'):
        base, ext = os.path.splitext(file_name)
        if not ext:
            ext = ".npz"
        file_name = base + ext

        hdlog.info("Loading Counter patterns from legacy file '%s'" % file_name)
        instance = cls()
        contents = np.load(file_name)
        instance._counts = dict(zip(contents['count_keys'], contents['count_values']))
        instance._patterns = contents['fp_list']
        instance._lookup_patterns = dict(zip(contents['lookup_fp_keys'], contents['lookup_fp_values']))
        instance._sequence = contents['sequence']
        contents.close()
        return instance

    # representation

    def __repr__(self):
        return '<Counter: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))


class PatternsRaw(Counter):
    # TODO: document
    _SAVE_TYPE = 'PatternsRaw'

    def __init__(self, patterns_raw=None, save_sequence=True):
        """
        Missing documentation
        
        Parameters
        ----------
        patterns_raw : Type, optional
            Description (default None)
        save_sequence : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
        super(PatternsRaw, self).__init__(counter=patterns_raw, save_sequence=save_sequence)

    # i/o

    def save(self, file_name='patterns_raw', extra=None):
        """
        Saves contents to file.

        Parameters
        ----------
        file_name : str, optional
            File name to save to (default 'patterns_raw')
        extra : dict, optional
            Extra information to save to file (default None)

        Returns
        -------
        Nothing
        """
        return super(PatternsRaw, self).save(file_name=file_name, extra=extra)

    @classmethod
    def load(cls, file_name='patterns_raw', load_extra=False):
        """
        Loads contents from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            patterns = PatternsRaw.load('file_name')

        Parameters
        ----------
        file_name : str, optional
            File name to load from (default 'patterns_raw')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        patterns : :class:`.PatternsRaw`
            Instance of :class:`.PatternsRaw` if loaded, `None` upon error
        """
        return super(PatternsRaw, cls).load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('Loading PatternsRaw patterns, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<PatternsRaw: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))


class PatternsHopfield(Counter):
    """
    record / counts of fixed-points of Hopfield network

    fixed_points / memories are stored in dictionary self.counts

    Parameters
        learner: hopfield network and learning params
        mtas: dict taking bin vects v to sums of orignal binary vectors converging to v

    Parameters
    ----------
    learner : Type, optional
        Description (default None)
    patterns_hopfield : Type, optional
        Description (default None)
    save_sequence : bool, optional
        Description (default True)
    save_raw : bool, optional
        Description (default True)

    Returns
    -------
    Value : Type
        Description
    """
    _SAVE_ATTRIBUTES_V1 = ['_counts', '_patterns', '_lookup_patterns',
                        '_sequence', '_skipped_patterns', '_seen_sequence',
                        '_mtas', '_mtas_raw']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'PatternsHopfield'

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
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._mtas

    @property
    def mtas_raw(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._mtas_raw

    def add_key(self, key, value=1, raw=None):
        """
        Missing documentation
        
        Parameters
        ----------
        key : Type
            Description
        value : int, optional
            Description (default 1)
        raw : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
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
        """
        combine your counts with another PatternsHopfield class
        
        Parameters
        ----------
        patterns_hopfield : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
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
        """
        M x N numpy array X as input (N neurons, M vects)
        
        Parameters
        ----------
        X : Type
            Description
        add_new : bool, optional
            Description (default True)
        rotate : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO warn if no network
        Y = self._learner.network(X)
        for x, y in zip(X, Y):
            self.chomp_vector(x, y, add_new=add_new, rotate=rotate)

    def chomp_vector(self, x, y, add_new=True, rotate=None):
        """
        stores bin vects (originally x) y and order of occurence
        
        Parameters
        ----------
        x : Type
            Description
        y : Type
            Description
        add_new : bool, optional
            Description (default True)
        rotate : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
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
        """
        Missing documentation
        
        Parameters
        ----------
        spikes : Type
            Description
        add_new : bool, optional
            Description (default True)
        window_size : int, optional
            Description (default 1)
        trials : Type, optional
            Description (default None)
        reshape : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
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
        return Spikes(spikes=Y)

    def pattern_to_mta_matrix(self, m):
        """
        Missing documentation
        
        Parameters
        ----------
        m : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        key = self._patterns[m]
        return self._mtas[key] / self._counts[key]

    def pattern_to_raw_patterns(self, m):
        """
        Missing documentation
        
        Parameters
        ----------
        m : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        key = self._patterns[m]
        return np.array(self._mtas_raw[key])

    def pattern_to_mta_std(self, m):
        """
        Missing documentation
        
        Parameters
        ----------
        m : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        raw_patterns = self.pattern_to_raw_patterns(m)
        return raw_patterns.std(axis=0)

    def top_mta_matrices(self, count):
        """
        finds (top count likely memory)-triggered averages
        
        Parameters
        ----------
        count : Type
            Description
        
        Returns
        -------
        Value : Type
            Description
        """
        top_mtas = []
        idx = np.array(self._counts.values()).argsort()[-count:]
        for i in idx:
            top_mtas.append(self.pattern_to_mta_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return np.array(top_mtas)

    def pattern_to_trial_raster(self, m, start=0, stop=None, trials=None):
        """
        Missing documentation
        
        Parameters
        ----------
        m : Type
            Description
        start : int, optional
            Description (default 0)
        stop : Type, optional
            Description (default None)
        trials : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        stop = stop or self._learner.spikes.M
        trials = trials or range(self._learner.spikes.T)
        key = self._patterns[m]
        sequence = np.array(self._sequence).reshape(self._learner.spikes.T,
                                                   self._learner.spikes.M - self._learner.window_size + 1)[:, start:stop]

        hits = (sequence == m).astype(int)
        return hits

    def approx_basin_size(self, max_corrupt_bits=1):
        """
        average bits corruption memory can stand
        
        Parameters
        ----------
        max_corrupt_bits : int, optional
            Description (default 1)
        
        Returns
        -------
        Value : Type
            Description
        """
        # TODO implement
        pass

    # i/o

    def save(self, file_name='patterns_hopfield', extra=None):
        """
        Saves contents to file.

        Parameters
        ----------
        file_name : str, optional
            File name to save to (default 'patterns_hopfield')
        extra : dict, optional
            Extra information to save to file (default None)

        Returns
        -------
        Nothing
        """
        super(PatternsHopfield, self).save(file_name=file_name, extra=extra)

    @classmethod
    def load(cls, file_name='patterns_hopfield', load_extra=False):
        """
        Loads contents from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            patterns = PatternsHopfield.load('file_name')

        Parameters
        ----------
        file_name : str, optional
            File name to load from (default 'patterns_hopfield')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        patterns : :class:`.PatternsHopfield`
            Instance of :class:`.PatternsHopfield` if loaded, `None` upon error
        """
        return super(PatternsHopfield, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        hdlog.debug('Loading PatternsHopfield patterns, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    @classmethod
    def load_legacy(cls, file_name='patterns_hopfield'):
        base, ext = os.path.splitext(file_name)
        if not ext:
            ext = ".npz"
        file_name = base + ext

        hdlog.info("Loading PatternHopfield patterns from legacy file '%s'" % file_name)
        instance = cls()
        contents = np.load(file_name)
        instance._counts = dict(zip(contents['count_keys'], contents['count_values']))
        instance._patterns = contents['fp_list']
        instance._lookup_patterns = dict(zip(contents['lookup_fp_keys'], contents['lookup_fp_values']))
        instance._sequence = contents['sequence']
        instance._mtas = dict(zip(contents['stas_keys'], contents['stas_values']))
        instance._sequence = contents['sequence']
        contents.close()
        return instance

    # representation

    def __repr__(self):
        return '<PatternsHopfield: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))


# end of source
