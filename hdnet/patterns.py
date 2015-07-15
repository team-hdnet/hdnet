# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.patterns
    ~~~~~~~~~~~~~~

    Record / counts of fixed-points of Hopfield network.

"""

from collections import OrderedDict

import os
import numpy as np
from hdnet.spikes import Spikes
from hdnet.util import hdlog, Restoreable


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
        Returns the counts of each pattern encountered in the
        raw data.
        
        Returns
        -------
        counts : dict
            Dictionary of counts of all patterns,
            indexed by pattern key
        """
        return self._counts

    @property
    def counts_by_label(self):
        """
        Returns the counts of each pattern encountered in the
        raw data.

        Returns
        -------
        counts : 1d numpy array, int
            Counts of all patterns, indexed by label
        """
        return [self._counts[p]
                for p in self.patterns]


    @property
    def patterns(self):
        """
        Returns the patterns encountered in the raw data
        as 1d vectors.
        
        Returns
        -------
        patterns : 2d numpy array, int
            Binary array of patterns encountered in the
            raw data, as 1d vectors
        """
        return self._patterns

    @property
    def num_patterns(self):
        """
        Returns the number of patterns encountered in the raw data.

        Returns
        -------
        N : int
            number of distinct patterns in the raw data
        """
        return len(self._patterns)

    @property
    def lookup_patterns(self):
        """
        Returns the lookup dictionary for the patterns,
        mapping a string representation of a pattern
        to a vector representation.
        
        Returns
        -------
        lookup : dict
            Lookup dictionary
        """
        return self._lookup_patterns

    @property
    def sequence(self):
        """
        Returns the sequence of patterns labels as encountered
        in the raw data. Pattern labels are allocated as integer
        numbers starting from 0 over the input data. Whenever
        a pattern was not encountered before, a new label is
        allocated.

        Returns
        -------
        sequence : 1d numpy array, int
            Sequence of pattern labels over raw data
        """
        return self._sequence

    @property
    def skipped_patterns(self):
        """
        Returns a binary vector signalling when a pattern
        was skipped due to rotation symmetry.
        
        Returns
        -------
        skipped : 1d numpy array
            Skipped patterns indicator
        """
        return self._skipped_patterns

    @property
    def seen_sequence(self):
        """
        Returns the sequence of seen flags for the
        patterns over the raw data. Each entry is
        binary and has a value of 1 if the pattern
        at this position occurred previously already.
        
        Returns
        -------
        seen : 1d numpy array
            Sequence of seen flags
        """
        return self._seen_sequence

    def __add__(self, other):
        """
        Merges counts of another  :class:`.Counter`
        object into this instance. Calls
        :meth:`.merge_counts`.
        
        Parameters
        ----------
        other : :class:`.Counter`
            Other Counter object
        
        Returns
        -------
        counter : :class:`.Counter`
            This instance
        """
        return self.merge_counts(other)

    def __len__(self):
        """
        Returns number of distinct patterns
        in this Counter.
        
        Returns
        -------
        length : int
            Number of distinct patterns
        """
        return len(self._counts.keys())

    def merge_counts(self, counter):
        """
        Merges counts of another :class:`.Counter`
        object into this instance.

        Parameters
        ----------
        other : :class:`.Counter`
            Other Counter object

        Returns
        -------
        counter : :class:`.Counter`
            This instance
        """
        for key in counter.counts.keys():
            key_ = Counter.key_for_pattern(Counter.pattern_for_key(key))
            self.add_key(key_, counter.counts[key])
        return self

    def add_key(self, key, value=1, **kwargs):
        """
        Adds a new key (pattern) to the collection.

        Parameters
        ----------
        key : str of '0', '1'
            Key of pattern to add, obtained from :meth:`Counter.key_for_pattern`
        value : int, optional
            Number of occurrences to add (default 1)
        raw : 2d numpy array, int, optional
            Raw pattern that converged to given memory (default None)

        Returns
        -------
        added : bool
            Flag whether key was previously known

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
        Counts patterns occurring as row vectors of N x M input
        matrix `X` and stores them. Calls :meth:`.chomp_vector`
        on row vectors of `X`.

        Parameters
        ----------
        X : M x N numpy array, int
            Binary source data.
        add_new : bool, optional
            Flag whether to store new memories (default True)
        rotate : tuple of length 2, int, optional
            Dimensions of window if patterns are to be
            collected modulo window rotations (default None)

        Returns
        -------
        Nothing
        """
        for x in X:
            self.chomp_vector(x, add_new=add_new, rotate=rotate)

    def chomp_spikes(self, spikes, add_new=True, window_size=1, trials=None, rotate=None):
        """
        Counts and stores patterns occurring over :class:`.Spikes`
        class using a sliding window of size `window_size`.

        Parameters
        ----------
        spikes : :class:`.Spikes`
            Instance of :class:`.Spikes` to operate on
        window_size : int, optional
            Window size to use (default 1)
        trials : int, optional
            Number of trials to use for reshape (default None)
        reshape : bool, optional
            Flag whether to reshape the spike vectors into
            matrix form before returning (default True)
        rotate : tuple of length 2, int, optional
            Dimensions of window if patterns are to be
            collected modulo window rotations (default None)

        Returns
        -------
        counter : :class:`.Counter`
            Returns pointer to itself
        """
        if rotate and add_new:
            self._skipped_patterns = 0
        X = spikes.to_windowed(window_size=window_size, trials=trials, reshape=True)
        self.chomp(X, add_new=add_new, rotate=rotate)
        return self

    def chomp_vector(self, x, add_new=True, rotate=None):
        """
        Counts occurrences of pattern in vector `x`, assigns it a
        integer label and stores it.

        Parameters
        ----------
        x : 1d numpy array, int
            Binary source vector.
        add_new : bool, optional
            Flag whether to store new memories (default True)
        rotate : tuple of length 2, int, optional
            Dimensions of window if patterns are to be
            collected modulo window rotations (default None)

        Returns
        -------
        bin_x, new_pattern, numrot : str, bool, int
            Key of pattern `x`, Flag whether pattern was seen
            before, number of rotations performed to obtain
            pattern identity (if `rotate` was given)
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

    def pattern_to_binary_matrix(self, key):
        """
        Returns a binary matrix representation of a pattern with the given key
        (as string of binary numbers).
        
        Parameters
        ----------
        key : str
            Key of pattern
        
        Returns
        -------
        pattern : 1d numpy array
            Representation of pattern as binary vector
        """
        key = self._patterns[key]
        return Counter.pattern_for_key(key)

    def top_binary_matrices(self, m):
        """
        Returns the top `m` likely patterns.
        
        Parameters
        ----------
        m : int
            Number of top likely patterns to return
        
        Returns
        -------
        patterns : numpy array
            `m` top likely patterns
        """
        top_binary = []
        idx = np.array(self._counts.values()).argsort()[-m:]
        for i in idx:
            top_binary.append(self.pattern_to_binary_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return top_binary


    def pattern_correlation_coefficients(self, labels = None, **kwargs):
        """
        Calculates the matrix of correlation coefficients between
        memories.

        Takes optional argument labels that allows to restrict the
        selection of patterns to a subset of all memories. Entries
        in labels have to be in the closed interval [0, self.num_patterns - 1].

        Parameters
        ----------
        labels : array_like, int
            Labels of patterns to consider
        kwargs : dictionary
            Additional arguments passed to np.corrcoef

        Returns
        -------
        C : 2d numpy array
            Matrix of normalized pairwise correlation coefficients
        """

        if labels is None:
            labels = xrange(self.num_patterns)

        pats = np.array([self.pattern_for_key(self._patterns[l]).ravel() for l in labels])
        return np.corrcoef(pats, **kwargs)

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

        # for t in xrange(stm_arr.shape[0]):
        #     arr[t * stm_arr.shape[1]:(t + 1) * stm_arr.shape[1]] = stm_arr[t]
        # 
        for c, pattern in enumerate(self._patterns):
            idx = (self._sequence == c)
            stim_avgs.append(stm_arr[idx].mean(axis=0))
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
        # internal function to load v1 file format
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
    """
    Catalogues binary vectors and their prevalence in raw spiking data.
    Subclass of :class:`.Counter`, extending its functionality.

    Parameters
    ----------
    patterns_raw : :class:`.PatternsRaw`, optional
        Patterns object to merge with (default None)
    save_sequence : bool, optional
        Flag whether to save the sequence of pattern
        labels, labels given by order of appearance (default True)

    Returns
    -------
    patterns : :class:`.PatternsRaw`.
        Instance of class :class:`.PatternsRaw`
    """
    _SAVE_TYPE = 'PatternsRaw'

    def __init__(self, patterns_raw=None, save_sequence=True):
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
        # internal function to load v1 file format
        hdlog.debug('Loading PatternsRaw patterns, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<PatternsRaw: {n} patterns ({u} unique)>'.format(n=sum(self._counts.values()), u=len(self._counts))


class PatternsHopfield(Counter):
    """
    Catalogues Hopfield fixed points of binary vectors
    and their prevalence in raw spiking data, optionally
    keeping references to the raw data.
    Subclass of :class:`.Counter`, extending its functionality.

    Parameters
    ----------
    learner : :class:`.Learner`, optional
        Learner instance to use that holds the underlying
        Hopfield Network (default None)
    patterns_hopfield : :class:`.PatternsHopfield`, optional
        Hopfield Patterns class to merge with (default None)
    save_sequence : bool, optional
        Flag whether to save the sequence of pattern
        labels with labels given by natural numbers in order
        of appearance (default True)
    save_raw : bool, optional
        Flag whether to save the raw patterns that converge
        to each memory under the Hopfield dynamics (default True)

    Returns
    -------
    patterns : :class:`.PatternsHopfield`.
        Instance of class :class:`.PatternsHopfield`
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
        Returns the memory triggered averages (MTAs) of all stored memories.

        Returns a Python dictionary keys of which are strings of binary digits
        representing the memory (the original memory can be obtained
        from the key using :meth:`Counter.pattern_for_key`) and values
        are 2d numpy arrays representing the memory triggered average.


        Returns
        -------
        mtas : dict
            Dictionary of MTAs of all stored memories
        """
        return self._mtas

    @property
    def mtas_raw(self):
        """
        Returns the set of all raw patterns encountered that converged
        to each stored memory. For each memory, the average of those
        patterns corresponds to its memory triggered average (MTA).

        Returns a Python dictionary keys of which are strings of binary digits
        representing the memory (the original memory can be obtained
        from the key using :meth:`Counter.pattern_for_key`) and values
        are lists of 2d numpy arrays representing raw patterns that converge
        to the given memory.

        Returns
        -------
        raw_patterns : dict
            Dictionary of lists of raw patterns converging to a given memory
        """
        return self._mtas_raw

    def add_key(self, key, value=1, raw=None):
        """
        Adds a new key (pattern) to the collection.
        
        Parameters
        ----------
        key : str of '0', '1'
            Key of memory to add, obtained from :meth:`Counter.key_for_pattern`
        value : int, optional
            Number of occurrences to add (default 1)
        raw : 2d numpy array, int, optional
            Raw pattern that converged to given memory (default None)
        
        Returns
        -------
        added : bool
            Flag whether key was previously known
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
        Combines counts with another PatternsHopfield class.
        
        Parameters
        ----------
        patterns_hopfield : :class:`.PatternsHopfield`
            Other :class:`.PatternsHopfield` class to merge
            counts with
        
        Returns
        -------
        patterns : :class:`.PatternsHopfield`
            Returns pointer to itself
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
        Computes Hopfield fixed points of M rows in N x M input
        matrix `X` using stored Hopfield network and stores
        the memories. Calls :meth:`.chomp_vector` on row
        vectors of `X`.
        The number of columns N has to equal the number
        of nodes in the underlying Hopfield network.

        Parameters
        ----------
        X : M x N numpy array, int
            Binary source data to converge.
        add_new : bool, optional
            Flag whether to store new memories (default True)
        rotate : tuple of length 2, int, optional
            Dimensions of window if patterns are to be
            collected modulo window rotations (default None)
        
        Returns
        -------
        Nothing
        """
        # TODO warn if no network
        Y = self._learner.network(X)
        for x, y in zip(X, Y):
            self.chomp_vector(x, y, add_new=add_new, rotate=rotate)

    def chomp_vector(self, x, y, add_new=True, rotate=None):
        """
        Associates binary raw data `x` with its Hopfield memory `y`,
        counting occurrences and storing raw data for the calculation
        of memory triggered averages (the average of all raw patterns
        in the data coverging to a given memory).

        Parameters
        ----------
        x : 1d numpy array, int
            Binary source data to converge.
        y : 1d numpy array, int
            Binary converged data.
        add_new : bool, optional
            Flag whether to store new memories (default True)
        rotate : tuple of length 2, int, optional
            Dimensions of window if patterns are to be
            collected modulo window rotations (default None)

        Returns
        -------
        Nothing
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

    def apply_dynamics(self, spikes, window_size=1, trials=None, reshape=True):
        """
        Computes Hopfield fixed points over data obtained from
        `spikes` using a sliding window of size `window_size`.

        Parameters
        ----------
        spikes : :class:`.Spikes`
            Instance of :class:`.Spikes` to operate on
        window_size : int, optional
            Window size to use (default 1)
        trials : int, optional
            Number of trials to use for reshape (default None)
        reshape : bool, optional
            Flag whether to reshape the spike vectors into
            matrix form before returning (default True)
        
        Returns
        -------
        spikes : :class:`.Spikes`
            Instance of spikes class with converged spikes
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

    def pattern_to_mta_matrix(self, label):
        """
        Returns the average of all raw patterns encountered that converged
        to a given stored memory pattern with label `label`. This average
        is called memory triggered average (MTA).
        
        Parameters
        ----------
        label : int
            Label of pattern to look up
        
        Returns
        -------
        mta : 1d numpy array
            MTA of memory with label `label`
        """
        key = self._patterns[label]
        return self._mtas[key] / self._counts[key]

    def pattern_to_raw_patterns(self, label):
        """
        Returns the list of all raw patterns encountered that converged
        to a given stored memory pattern with label `label`.

        Parameters
        ----------
        label : int
            Label of pattern to look up

        Returns
        -------
        raw : list of 1d numpy array
            Raw patterns converging to memory with label `label`
        """
        key = self._patterns[label]
        return np.array(self._mtas_raw[key])

    def pattern_to_mtv(self, m):
        """
        Returns the element-wise variance of each position in a pattern
        across all raw patterns encountered that converged to a given
        stored memory pattern with label `label`. This is called the
        *Memory Triggered Variance* (MTV). It is a meansure for how
        diverse the underlying patterns are converging to the same
        memory and can be seen as a proxy for the basin size of
        that memory (i.e. fixed point) under the given Hopfield dynamics.

        Parameters
        ----------
        label : int
            Label of pattern to look up

        Returns
        -------
        mtv : 1d numpy array
            Element-wise variance
        """
        raw_patterns = self.pattern_to_raw_patterns(m)
        return raw_patterns.var(axis=0)

    def top_mta_matrices(self, count):
        """
        Returns a list of memory triggered averages (MTAs) of the
        memories occurring the most in the encountered data.
        
        Parameters
        ----------
        count : int
            Number of mostly occurring memories to consider
        
        Returns
        -------
        mtas : 2d numpy array
            Array of MTAs belonging to top occurring memories
        """
        top_mtas = []
        idx = np.array(self._counts.values()).argsort()[-count:]
        for i in idx:
            top_mtas.append(self.pattern_to_mta_matrix(self._lookup_patterns[self._counts.keys()[i]]))
        return np.array(top_mtas)

    def pattern_to_trial_raster(self, label, start=0, stop=None, trials=None):
        """
        Returns binary matrix signalling when the memory with the given
        label `label` apprears in the data.
        
        Parameters
        ----------
        label : int
            Label of pattern to look up
        start : int, optional
            First index in each trial (default 0)
        stop : int, optional
            Last index in each trial, if None
             whole trial will be used (default None)
        trials : int, optional
            Number of trials, if None taken from
            underlying :class:`.Learner` class (default None)
        
        Returns
        -------
        hits : 2d numpy array
            Binary array with a value of 1 encoding the occurrence
            of the pattern with the given label `label`
        """
        stop = stop or self._learner.spikes.M
        trials = trials or range(self._learner.spikes.T)
        key = self._patterns[label]
        sequence = np.array(self._sequence).reshape(self._learner.spikes.T,
                                                   self._learner.spikes.M - self._learner.window_size + 1)[:, start:stop]

        hits = (sequence == label).astype(int)
        return hits

    def approximate_basin_size(self, max_corrupt_bits=1):
        """
        Average bits corruption a memory can stand.

        .. note:

            Not implemented yet
        
        Parameters
        ----------
        max_corrupt_bits : int, optional
            Maximal number of corrupted bits to try (default 1)
        
        Returns
        -------
        basin_sizes : numpy array
            Approximated basin sizes of each memory
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
        # internal function to load v1 file format
        hdlog.debug('Loading PatternsHopfield patterns, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    @classmethod
    def load_legacy(cls, file_name='patterns_hopfield'):
        # internal function to load legacy file format
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
