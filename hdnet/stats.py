# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.stats
    ~~~~~~~~~~~

    Statistics module.

"""

__version__ = "0.1"

import numpy as np
from util import hdlog



class SequenceAnalyzer(object):
    """
    Calculates various aspects of sequences.

    Parameters
    ----------
    counter : instance of :class:`.Counter`
        Base counter the sequence to operate on

    Returns
    -------
    analyzer : instance of :class:`.SequenceAnalyzer`
        Instance of :class:`.SequenceAnalyzer`
    """

    def __init__(self, counter):
        self._counter = counter
        self._sequence = counter.sequence.copy()
        self._markov_probabilities = None
        self._markov_entropies = None
        self._label_probabilities = None
        self._sequence = counter.sequence
        self._graph = None

    @property
    def counter(self):
        """
        Returns the :class:`.Counter` object this class
        operates.

        Returns
        -------
        counter : instance of :class:`.Counter`
        """
        return self._counter

    @property
    def sequence(self):
        """
        Returns the (possibly filtered) sequence this class
        currently operates on.

        Returns
        -------
        sequence : 1d numpy array, int
            Sequence of integer labels
        """
        return self._sequence

    @property
    def label_probabilities(self):
        """
        Returns probabilities of labels in sequence,
        see :meth:`compute_label_probabilities`.

        Returns
        -------
        prob : 1d numpy array, float
            Vector of label probabilities
        """
        return self.compute_label_probabilities()

    @property
    def label_markov_probabilities(self):
        """
        Returns Markov transition probabilities of labels in sequence,
        see :meth:`compute_label_markov_probabilities`.

        Returns
        -------
        markov_prob : 2d numpy array, float
            Matrix of Markov transition probabilities
        """
        return self.compute_label_markov_probabilities()

    @property
    def label_markov_entropies(self):
        """
        Returns entropies of Markov transition probabilities
        of labels in sequence, see
        :meth:`compute_label_markov_entropies`.

        Returns
        -------
        markov_ent : 1d numpy array, float
            Vector of entropies computed from Markov transition probabilities
        """
        return self.compute_label_markov_entropies()

    @property
    def markov_graph(self):
        """
        Returns Markov graph belowing to sequence of labels in sequence, see
        :meth:`compute_markov_graph`.

        Returns
        -------
        markov_graph : networkx.DiGraph instance
            Directed graph with edge weights corresponding to Markov transition
            probabilities.
        """
        return self.compute_markov_graph()

    def filter_sequence_repeating_labels(self, repetitions=2, sequence=None):
        """
        Removes all consecutive repetitions occuring more than repetitions times
        (default: 2) from sequence

        Parameters
        ----------
        sequence : Type
            Description
        repetitions : int, optional
            Description (default 2)

        Returns
        -------
        Value : Type
            Description
        """
        if sequence is None:
            sequence = self.sequence

        seq_filtered = []
        for i in xrange(repetitions, len(sequence)):
            if all([x == sequence[i] for x in sequence[i - repetitions:i]]):
                continue
            seq_filtered.append(sequence[i])

        seq_filtered = np.array(seq_filtered)
        if sequence is self.sequence:
            self._sequence = seq_filtered
        return seq_filtered

    def compute_label_occurrences(self, sequence=None):
        """
        Compute occurrences of labels in sequence

        Parameters
        ----------
        sequence : Type
            Description

        Returns
        -------
        Value : Type
            Description
        """
        from collections import Counter
        counter = Counter(sequence)
        return counter.items()

    def filter_sequence_threshold(self, threshold, replacement_label=-1, sequence=None):
        """
        Filter out all labels from sequence, occuring less than `threshold`
        times and replace them with `replacement_label`.

        Parameters
        ----------
        sequence : Type
            Description
        repetitions : int, optional
            Description (default 2)

        Returns
        -------
        Value : Type
            Description
        """
        if sequence is None:
            sequence = self.sequence

        from collections import Counter
        counter = Counter(sequence)
        indices_sorted = np.argsort(np.array(counter.values()))[::-1]
        keys = counter.keys()

        filter_at = len(indices_sorted)
        for i in indices_sorted:
            if counter[i] < threshold:
                filter_at = i
                break

        for i in xrange(filter_at, len(indices_sorted)):
            sequence[sequence == keys[indices_sorted[i]]] = replacement_label

        return sequence

    def filter_sequence_top_occurring(self, count, replacement_label=-1, sequence=None):
        """
        Filter out all labels from sequence, occuring less than `threshold`
        times and replace them with `replacement_label`.

        Parameters
        ----------
        sequence : Type
            Description
        repetitions : int, optional
            Description (default 2)

        Returns
        -------
        Value : Type
            Description
        """
        if sequence is None:
            sequence = self.sequence

        from collections import Counter
        counter = Counter(sequence)
        indices_sorted = np.argsort(np.array(counter.values()))[::-1][count:]
        keys = counter.keys()

        for i in indices_sorted:
            sequence[sequence == keys[i]] = replacement_label

        return sequence

    def compute_label_probabilities(self, sequence=None, parent=None):
        """
        Compute probability vector of patterns as empirical probabilities.
        If parent counter object present then return probabilty vector in
        that space.
        
        Parameters
        ----------
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)
        parent : :class:`.Counter`, optional
            Parent Counter object (default None)
        
        Returns
        -------
        probabilities : numpy array
            Vector of label probabilities
        """
        if sequence is None:
            if self._label_probabilities is not None:
                return self._label_probabilities
            sequence = self.sequence

        labels = list(set(sequence))
        if parent is not None:
            values = np.zeros(len(parent.counts))
            for i in xrange(len(labels)):
                if labels[i] in parent.counts:
                    values[parent.lookup_patterns[labels[i]]] = \
                        self.counter.counts[labels[i]]
        else:
            values = np.array([self.counter.counts[l] for l in labels])

        probs = 1. * values / values.sum()

        if parent is None and sequence is self.sequence:
            self._label_probabilities = probs

        return probs

    def compute_label_markov_probabilities(self, sequence=None):
        """
        Computes matrix of Markov transition probabilities of all labels
        occuring in `sequence` over this sequence.

        Parameters
        ----------
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)

        Returns
        -------
        markov_prob : 2d numpy array
            Matrix of markov transition probabilities of entries in `sequence`
        """
        if sequence is None:
            if self._markov_probabilities is not None:
                return self._markov_probabilities
            sequence = self.sequence

        sequence = np.atleast_1d(sequence)
        labels = list(set(sequence))
        n_fp = len(set(sequence))
        m_prob = np.zeros((n_fp, n_fp), dtype=float)

        for p in xrange(n_fp):
            for pidx in np.where(sequence == labels[p])[0]:
                if pidx == len(sequence) - 1:
                    continue
                m_prob[p, labels.index(sequence[pidx + 1])] += 1
            m_prob[p, :] /= m_prob[p, :].sum()

        if sequence is self.sequence:
            self._markov_probabilities = m_prob
        return m_prob

    def compute_label_markov_entropies(self, markov_probabilities=None, eps=1e-12):
        """
        Computes the entropy of each label using its Markov transition
        probabilities in the space of all labels.

        Parameters
        ----------
        markov_probabilities : 2d numpy array, float, optional
            Markov transtion matrix to use, if None
            :meth:`label_markov_probabilities` is used (default None)
        eps : int, optional
            Threshold value below which a float is assumed to be 0
            (default 1e-12)

        Returns
        -------
        markov_entropies : 1d numpy array, float
            Vector of entropies calculated over Markov transition probabilities
        """
        if markov_probabilities is None:
            if self._markov_entropies is not None:
                return self._markov_entropies
            markov_probabilities = self.label_markov_probabilities

        h = lambda a: -np.sum(a * np.log2(a))
        entropies = np.abs(np.array([h(x[x > eps]) for x in markov_probabilities.copy()]))

        if markov_probabilities is self.label_markov_probabilities:
            self._markov_entropies = entropies
        return entropies

    def entropy(self):
        """
        Computes entropy over probability distribution of sequence of
        pattern labels.

        Returns
        -------
        entropy : float
            Entropy of probability distribution
        """
        probs = self.label_probabilities()
        return -(probs * np.log2(probs)).sum()

    def compute_markov_graph(self, markov_probabilities=None, node_labels=None,
                             thres=0, no_loop=False):
        """
        Computes the directed state transition graph over all labels using the
        package NetworkX. Each directed edge (x, y) is assigned as weight the
        Markov transition probability from label x to label y.
        markov_probabilities: Markov transition probabilities of labels,
        2d numpy array
        node_labels: remapped labels (optional, default None)
        thres: weight threshold for edges; only edges above that weight
        are included in the graph (default 0)
        no_loop: boolean flag specifiying handling of self-loops; if set
        to True, self-loops are discarded.
        NetworkX DiGraph with lables as nodes and Markov transition
        probabilities as edges

        .. note::

            This function needs the `networkx` package.

        Parameters
        ----------
        markov_probabilities : Type
            Description
        node_labels : Type, optional
            Description (default None)
        thres : int, optional
            Description (default 0)
        no_loop : bool, optional
            Description (default False)

        Returns
        -------
        markov_graph : networkx.DiGraph instance
            Directed graph with edge weights corresponding to Markov transition
            probabilities.
        """
        import networkx as nx

        if markov_probabilities is None:
            if self._graph is not None:
                return self._graph
            markov_probabilities = self.label_markov_probabilities

        # nodes
        n = len(markov_probabilities)
        if node_labels is None:
            node_labels = list(set(self.sequence))

        # edges
        edges = []
        for i in xrange(n):
            for j in xrange(n):
                if i == j and no_loop:
                    continue
                if markov_probabilities[i][j] > thres:
                    k = node_labels[i]
                    l = node_labels[j]
                    edges.append((k, l, {'weight': markov_probabilities[i][j]}))

        # construct graph
        transition_graph = nx.DiGraph()
        transition_graph.add_nodes_from(node_labels)
        transition_graph.add_edges_from(edges)

        if markov_probabilities is self.label_markov_probabilities:
            self._graph = transition_graph

        return transition_graph

    def reduce_graph_self_loops(self, g=None):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        removed = 0
        for n in g.nodes():
            if g.has_edge(n, n):
                g.remove_edge(n, n)
                removed += 1
        return removed

    def reduce_graph_brute(self, filtered_nodes, g=None):
        """
        Missing documentation

        Parameters
        ----------
        g : Type
            Description
        filtered_nodes : Type
            Description

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        # remove all nodes not in filtered_nodes
        for n in g.nodes():
            if not n in filtered_nodes:
                g.remove_node(n)
        return g

    def reduce_graph_bridge(self, g=None):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        # reduce bridge at v: u -> v -> w
        # becomes u -> w
        removed = 1
        while removed > 0:
            removed = 0
            for n in g.nodes():
                pred = g.predecessors(n)
                succ = g.successors(n)
                if len(pred) == 1 and len(succ) == 1:
                    # print "removing", succ[0]
                    weight = float(np.mean([g.get_edge_data(pred[0],n)['weight'], g.get_edge_data(n,succ[0])['weight']]))
                    g.add_edge(pred[0], succ[0], weight=weight)
                    g.remove_node(n)
                    removed += 1
        return g

    def reduce_graph_stub(self, g=None):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        # recursively remove all stub nodes with no successors
        removed = 1
        while removed > 0:
            removed = 0
            for n in g.nodes():
                if len(g.predecessors(n)) + len(g.successors(n)) <= 1:
                        g.remove_node(n)
                        removed += 1
        return g

    def reduce_graph_loop(self, g):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        # remove all loop nodes v, i.e. u -> v -> u
        removed = 1
        while removed > 0:
            removed = 0
            for n in g.nodes():
                pred = g.predecessors(n)
                succ = g.successors(n)
                if len(pred) == 1 and len(succ) == 1 and pred == succ:
                    g.remove_node(n)
                    removed += 1
        return g

    def reduce_graph_nloop(self, g, n, node):
        """
        Missing documentation

        Parameters
        ----------
        g : Type
            Description
        n : Type
            Description
        node : Type
            Description

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        import networkx as nx

        pred = g.predecessors(node)
        for p in pred:
            for path in nx.all_simple_paths(g, node, p):
                if len(path) >= n or not g.has_edge(path[-1], node):
                    continue
                g.remove_edge(path[-1], node)

    def reduce_graph_triangles(self, g=None):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        import itertools
        removed = 1
        while removed > 0:
            removed = 0
            for n1 in g.nodes():
                for n2, n3 in itertools.combinations(g.successors(n1), 2):
                    edges = [(n1, n2), (n1, n3), (n2, n3)]
                    #print edges
                    if not all([g.has_edge(*e) for e in edges]):
                        #print 'skipped'
                        continue
                    m = np.argmin([g.get_edge_data(*e)['weight'] for e in edges])
                    #print 'remove ', edges[m]
                    g.remove_edge(*edges[m])
                    removed += 1
        return removed

    def reduce_graph_out_degree(self, thres, g=None):
        """
        Missing documentation

        Parameters
        ----------
        g : Type
            Description
        thres : Type
            Description

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        removed = 0
        for n in g.nodes():
            if g.out_degree(n) > thres:
                g.remove_node(n)
                removed += 1
        return removed

    def calculate_loops_entropy_scores(self, g, n, node_entropies, min_len=5, max_len=20, weighting=None, weighting_element=None):
        """
        Calculate entropy scores of loops (simple closed paths) in g starting and
        terminating in given node n. An entropy score of a path is the sum of the
        entropies (as specified in node_entropies) of each node contained in the path.
        g: NetworkX graph
        n: base node
        node_entropies: 1d array of node entropies
        (loops, scores), where loops is a 1d array of loops and scores a
        1d array of loop scores. Index in scores identical to index in loops,
        arrays sorted by score (ascending).

        Parameters
        ----------
        g : Type
            Description
        n : Type
            Description
        node_entropies : Type
            Description
        min_len : int, optional
            Description (default 5)
        max_len : int, optional
            Description (default 20)
        weighting : Type, optional
            Description (default None)
        weighting_element : Type, optional
            Description (default None)

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        import networkx as nx
        loops = []
        for p in g.predecessors(n):
            loops.extend([p + [n] for p in nx.all_simple_paths(g, n, p, cutoff=max_len) if len(p) >= min_len - 1])

        if weighting is None:
            weighting = lambda x: 1. / len(x)
        if weighting_element is None:
            weighting_element = lambda x, p: x
        scores = np.array([sum([weighting_element(node_entropies[m], i) for i, m in enumerate(path)]) * weighting(path) for path in loops])
        sort = np.argsort(scores)

        return np.array(loops)[sort], scores[sort]

    def calculate_paths_entropy_scores(self, g, n1, n2, node_entropies, min_len=5, max_len=20):
        """
        Calculate entropy scores of all simple paths in g from n1 to n2.
        An entropy score of a path is the sum of the entropies (as specified in
        node_entropies) of each node contained in the path.
        g: NetworkX graph
        n1: start node
        n2: end node
        node_entropies: 1d array of node entropies
        (paths, scores), where paths is a 1d array of paths and scores a
        1d array of path scores. Index in scores identical to index in paths,
        arrays sorted by score (ascending).

        Parameters
        ----------
        g : Type
            Description
        n1 : Type
            Description
        n2 : Type
            Description
        node_entropies : Type
            Description
        min_len : int, optional
            Description (default 5)
        max_len : int, optional
            Description (default 20)

        Returns
        -------
        Value : Type
            Description
        """
        if g is None:
            g = self.markov_graph
        import networkx as nx
        paths = [ p for p in nx.all_simple_paths(g, n1, n2, cutoff=max_len) if len(p) >= min_len]
        scores = np.array([sum([node_entropies[m] for m in path]) / 1. * len(path) for path in paths])
        sort = np.argsort(scores)

        return np.array(paths)[sort], scores[sort]

    @staticmethod
    def subseqs(sequence, length):
        """
        Enumerates all subsequences of given `length`
        in `sequence`. Lazy, returns generator object.

        Parameters
        ----------
        sequence : list or array
            Sequence to enumerate subsequences for
        length : int
            Length of subsequences to enumerate

        Returns
        -------
        generator : generator object
            Lazy generator object for subsequences
        """
        for i in xrange(len(sequence) - length + 1):
            yield sequence[i:i + length]

    def find_subsequences(self, thresholds, sequence=None):
        """
        Enumerates all subsequences of length `len(thresholds)`
        in `sequence` (if sequence is `None` the possibly
        filtered sequence from the stored counter object is taken).
        Subsequences of length i are only considered if they
        appear at least `thresholds[i - 1]` times in the sequence.

        Parameters
        ----------
        thresholds : list, int
            List of threshold values
        sequence : list or numpy array, optional
            Sequence to consider, if `None` defaults to stored
            sequence (default `None`)

        Returns
        -------
        sequences : list of dicts
            List of dictionaries containing all found sequences
            as keys and counts as values. Keys are labels separated
            by ','.
        """
        if sequence is None:
            sequence = self.sequence

        import collections
        counts = {str(item[0]): item[1] for item in collections.Counter(self.sequence).items()}

        all_counts = []

        maxlen = len(thresholds)
        for l in xrange(2, maxlen + 2):
            for k in counts.keys():
                if counts[k] < thresholds[l - 2]:
                    del counts[k]

            all_counts.append(counts)

            if l == maxlen + 1:
                break

            hdlog.info('processing subsequences of length %d' % l)
            counts_new = {}
            for s in SequenceAnalyzer.subseqs(sequence, l):
                subkey = ','.join([str(x) for x in s[:l - 1]])
                if subkey not in counts:
                    continue
                key = subkey + ',' + str(s[l - 1])
                if key not in counts_new:
                    counts_new[key] = 1
                else:
                    counts_new[key] += 1

            counts = counts_new

        return all_counts

    def find_subsequences_positions(self, subsequence, sequence=None):
        """
        Enumerates all positions of given `subsequence`
        in `sequence` (if sequence is `None` the possibly
        filtered sequence from the stored counter object is taken).

        Parameters
        ----------
        subsequence : list or numpy array
            Subsequence to search for
        sequence : list or numpy array, optional
            Sequence to consider, if `None` defaults to stored
            sequence (default `None`)

        Returns
        -------
        positions : 1d numpy array
            List of positions in `sequence` where `subsequence` occurs
        """
        if sequence is None:
            sequence = self.sequence

        positions = []
        subsequence = np.atleast_1d(subsequence)
        for i in xrange(len(subsequence), len(sequence)):
            if np.all(sequence[i - len(subsequence):i] == subsequence):
                positions.append(i)

        return np.array(positions)


# end of source
