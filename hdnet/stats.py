# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.stats
    ~~~~~~~~~~~

    Statistics module. Contains functions for analyzing sequences of
    memories and miscellaneous statistics functions.

"""

from __future__ import print_function

import numpy as np
from hdnet.util import hdlog


class SequenceAnalyzer(object):
    """
    Analyzes various aspects of sequences of memory labels, such
    as label probabilities, frequent sub-sequences and Markov
    transition probabilities.

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
        self._sequence = np.array(counter.sequence).copy()
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
        Removes all consecutive repetitions of labels from sequence
        occurring more than `repetitions` times (default: 2).

        Parameters
        ----------
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)
        repetitions : int, optional
            Description (default 2)

        Returns
        -------
        filtered_sequence : 1d numpy array
            Filtered sequence
        """
        if sequence is None:
            sequence = self.sequence

        seq_filtered = []
        for i in range(repetitions, len(sequence)):
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
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)

        Returns
        -------
        occurrences : dict, label => number of occurrences
            Number of occurrences for all labels
        """
        if sequence is None:
            sequence = self.sequence
        from collections import Counter
        counter = Counter(sequence)
        return counter.items()

    def filter_sequence_threshold(self, threshold, replacement_label=-1, sequence=None):
        """
        Filter out all labels from sequence, occurring less than `threshold`
        times and replace them with `replacement_label`.

        Parameters
        ----------
        threshold : int
            Minimal number of occurrences to not filter out label
        replacement_label : int, optional
            Replacement label used for a label that is dropped (default -1)
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)

        Returns
        -------
        filtered_sequence : 1d numpy array
            Filtered sequence
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

        for i in range(filter_at, len(indices_sorted)):
            sequence[sequence == keys[indices_sorted[i]]] = replacement_label
        return sequence

    def filter_sequence_top_occurring(self, count, replacement_label=-1, sequence=None):
        """
        Filter out all labels from sequence, occuring less than `threshold`
        times and replace them with `replacement_label`.

        Parameters
        ----------
        count : int
            Number of top occurring labels to keep
        replacement_label : int, optional
            Replacement label used for a label that is dropped (default -1)
        sequence : 1d numpy array, int, optional
            Sequence of symbols to consider, if None :meth:`sequence` is
            used (default None)

        Returns
        -------
        filtered_sequence : 1d numpy array
            Filtered sequence
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

    @staticmethod
    def subseqs(sequence, length):
        """
        Enumerates all subsequences of given `length`
        in `sequence`. Lazy, returns generator object.

        Parameters
        ----------
        sequence : list or numpy array
            Sequence to enumerate subsequences for
        length : int
            Length of subsequences to enumerate

        Returns
        -------
        generator : generator object
            Lazy generator object for subsequences
        """
        for i in range(len(sequence) - length + 1):
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
            as keys and counts as values. Keys are memory labels separated
            by ','.
        """
        if sequence is None:
            sequence = self.sequence

        import collections
        counts = {str(item[0]): item[1] for item in collections.Counter(self.sequence).items()}

        all_counts = []

        maxlen = len(thresholds)
        for l in range(2, maxlen + 2):
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
        for i in range(len(subsequence), len(sequence)):
            if np.all(sequence[i - len(subsequence):i] == subsequence):
                positions.append(i)

        return np.array(positions)

    def compute_label_probabilities(self, sequence=None, parent=None):
        """
        Compute probability vector of patterns as empirical probabilities.
        If parent counter object present then return probability vector in
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
            for i in range(len(labels)):
                if labels[i] in parent.counts:
                    values[parent.lookup_patterns[labels[i]]] = \
                        self.counter.counts_by_label[labels[i]]
        else:
            values = np.array([self.counter.counts_by_label[l] for l in labels])

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

        for p in range(n_fp):
            for pidx in np.where(sequence == labels[p])[0]:
                if pidx == len(sequence) - 1:
                    continue
                m_prob[p, labels.index(sequence[pidx + 1])] += 1
            m_prob[p, :] /= m_prob[p, :].sum()

        if sequence is self.sequence:
            self._markov_probabilities = m_prob
        return m_prob

    def compute_label_markov_entropies(self, markov_probabilities=None, eps=1e-9):
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
            (default 1e-9)

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
        probs = self.label_probabilities
        return -(probs * np.log2(probs)).sum()

    def compute_markov_graph(self, markov_probabilities=None, node_labels=None,
                             thres=0, no_cycle=False):
        """
        Computes the directed state transition graph over all labels using the
        package NetworkX. Each directed edge (x, y) is assigned as weight the
        Markov transition probability from label x to label y.
        markov_probabilities:
        2d numpy array
        node_labels: remapped labels (optional, default None)
        thres: weight threshold for edges; only edges above that weight
        are included in the graph (default 0)
        no_cycle: boolean flag specifiying handling of self-cycles; if set
        to True, self-cycles are discarded.
        NetworkX DiGraph with lables as nodes and Markov transition
        probabilities as edges

        .. note::

            This function needs the `networkx` package.

        Parameters
        ----------
        markov_probabilities : 2d numpy array, optional
            Markov transition probabilities of labels, if None
            Markov transition probabilities of internal sequence are used
            (default None)
        node_labels : list or 1d numpy array, optional
            Node labels to use, if None labels of internal sequence
            are used (default None)
        thres : int, optional
            Threshold to exclude nodes occurring less than
            a given number of times in the sequence (default 0)
        no_cycle : bool, optional
            Flag whether to not include self-cycles at nodes (default False)

        Returns
        -------
        markov_graph : networkx.DiGraph instance
            Directed graph with edge weights corresponding to Markov transition
            probabilities.
        """
        import networkx as nx

        if self._graph is not None:
            return self._graph

        if markov_probabilities is None:
            markov_probabilities = self.label_markov_probabilities

        # nodes
        n = len(markov_probabilities)
        if node_labels is None:
            node_labels = list(set(self.sequence))

        # edges
        edges = []
        for i in range(n):
            for j in range(n):
                if i == j and no_cycle:
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

    def reduce_graph_self_cycles(self, graph=None):
        """
        Removes self cycles u -> u of all nodes u in `graph`.

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        removed = 0
        for n in graph.nodes():
            if graph.has_edge(n, n):
                graph.remove_edge(n, n)
                removed += 1
        return removed

    def reduce_graph_brute(self, filtered_nodes, graph=None):
        """
        Removes self cycles u -> u of all nodes u in `graph`.

        Parameters
        ----------
        filtered_nodes : Type
            Description
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        # remove all nodes not in filtered_nodes
        removed = 0
        for n in graph.nodes():
            if not n in filtered_nodes:
                graph.remove_node(n)
                removed += 1
        return graph, removed

    def reduce_graph_bridge(self, graph=None):
        """
        Removes all "bridge" nodes v from `graph`, where a bridge node
        is defined as one having only one incoming and only one outgoing
        edge, i.e. u -> v -> w.

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        # reduce bridge at v: u -> v -> w
        # becomes u -> w
        all_removed = []
        removed = 1
        while removed > 0:
            removed = 0
            for n in graph.nodes():
                pred = graph.predecessors(n)
                succ = graph.successors(n)
                if len(pred) == 1 and len(succ) == 1:
                    weight = float(np.mean([graph.get_edge_data(pred[0], n)['weight'], 
                                            graph.get_edge_data(n,succ[0])['weight']]))
                    graph.add_edge(pred[0], succ[0], weight=weight)
                    graph.remove_node(n)
                    removed += 1
                    all_removed.append(n)
        return all_removed

    def reduce_graph_stub(self, graph=None):
        """
        Removes all "stub" nodes v from `graph`, where a stub node
        is defined as one having only only one incoming and no 
        outgoing edges, i.e. u -> v.

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        # recursively remove all stub nodes with no successors
        all_removed = []
        removed = 1
        while removed > 0:
            removed = 0
            for n in graph.nodes():
                if len(graph.predecessors(n)) + len(graph.successors(n)) <= 1:
                        graph.remove_node(n)
                        removed += 1
                        all_removed.append(n)
        return all_removed

    def reduce_graph_cycle(self, graph=None):
        """
        Removes all "cycle" nodes v from `graph`, where a cycle node
        is defined as one having only one incoming and only one outgoing
        edge, to the same node, i.e. u -> v -> u.

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        # remove all cycle nodes v, i.e. u -> v -> u
        all_removed = []
        removed = 1
        while removed > 0:
            removed = 0
            for n in graph.nodes():
                pred = graph.predecessors(n)
                succ = graph.successors(n)
                if len(pred) == 1 and len(succ) == 1 and pred == succ:
                    graph.remove_node(n)
                    removed += 1
                    all_removed.append(n)
        return all_removed

    def reduce_graph_ncycle(self, node, n, graph=None):
        """
        Removes all edges from `graph`, that belong to simple
        closed paths (i.e. cycles) around the given `node`
        that do not have at least length `n`.

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Parameters
        ----------
        node : str
            Label of starting node
        n : Type
            Minimal path length
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        import networkx as nx

        removed = []
        pred = graph.predecessors(node)
        for p in pred:
            for path in nx.all_simple_paths(graph, node, p):
                if len(path) >= n or not graph.has_edge(path[-1], node):
                    continue
                graph.remove_edge(path[-1], node)
                removed.append((path[-1], node))
        return removed
                

    def reduce_graph_triangles(self, graph=None):
        """
        Removes all triangles from `graph` (by deleting edges with the
        lowest weight) of the form u -> v, u -> w, v -> w. Deletes
        the edge with the lowest weight to delete the triangle (u, v, w).

        Parameters
        ----------
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        import itertools
        all_removed = []
        removed = 1
        while removed > 0:
            removed = 0
            for n1 in graph.nodes():
                for n2, n3 in itertools.combinations(graph.successors(n1), 2):
                    edges = [(n1, n2), (n1, n3), (n2, n3)]
                    if not all([graph.has_edge(*e) for e in edges]):
                        continue
                    m = np.argmin([graph.get_edge_data(*e)['weight'] for e in edges])
                    graph.remove_edge(*edges[m])
                    removed += 1
                    all_removed.append((edges[m][0], edges[m][1]))
        return all_removed

    def reduce_graph_out_degree(self, thres_max, thres_min=1, graph=None):
        """
        Removes all nodes from `graph` that have an out-degree
        of more than `thres_max` or an out-degree of less than
        `thres_min`.

        Parameters
        ----------
        thres_max : int
            Maximal out degree to retain
        thres_min : int, optional
            Minimal out degree to retain (default 1)
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        n_removed : int
            Number of removed nodes
        """
        if graph is None:
            graph = self.markov_graph
        removed = []
        for n in graph.nodes():
            if graph.out_degree(n) < thres_min or \
                    graph.out_degree(n) > thres_max:
                graph.remove_node(n)
                removed.append(n)
        return removed
    
    def calculate_cycles_entropy_scores(self, node, min_len=2, max_len=20, weighting=None,
                                       weighting_element=None, node_entropies=None, graph=None):
        """
        Calculate entropy scores of cycles (simple closed paths) in `graph` starting and
        terminating in given node `node`. An entropy score of a path is a weighted  sum
        of the entropies of each node contained in the path.

        Parameters
        ----------
        node : str
            Label of base node
        min_len : int, optional
            Minimal length of cycle to consider (default 2)
        max_len : int, optional
            Maximal length of cycle to consider (default 20)
        weighting : Type, optional
            Weighting function, if None `lambda x: 1./len(x)` is taken
            (default None)
        weighting_element : Type, optional
            Weighting function per element, if None `lambda x, p: x`
            is taken (default None)
        node_entropies : 1d numpy array, optional
            Node entropies to use, if None entropies from Markov 
            transition probabilities of internal sequence are used 
            (default None)
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        (cycles, scores) : (1d numpy array, 1d numpy array) 
            Scored cycles, where cycles is a 1d array of cycles and scores a
            1d array of cycle scores. Index in scores identical to index in cycles,
            arrays sorted by score (ascending).
        """
        if graph is None:
            graph = self.markov_graph
        if node_entropies is None:
            node_entropies = self.label_markov_entropies
        import networkx as nx
        cycles = []
        for p in graph.predecessors(node):
            cycles.extend(
                [p + [node] for p in nx.all_simple_paths(graph, node, p, cutoff=max_len) 
                 if len(p) >= min_len - 1])

        if weighting is None:
            weighting = lambda x: 1. / len(x)
        if weighting_element is None:
            weighting_element = lambda x, p: x
        scores = np.array(
            [sum([weighting_element(node_entropies[m], i) for i, m in enumerate(path)]) * weighting(path)
             for path in cycles])
        sort = np.argsort(scores)
        return np.array(cycles)[sort], scores[sort]

    def calculate_paths_entropy_scores(self, node1, node2, min_len=2, max_len=20, weighting=None,
                                       weighting_element=None, node_entropies=None, graph=None):
        """
        Calculate entropy scores of all simple paths in `graph` from `node1` to `node2`.
        An entropy score of a path is a weighted sum of the entropies of each node contained
        in the path.

        Parameters
        ----------
        node : str
            Label of base node
        min_len : int, optional
            Minimal length of cycle to consider (default 2)
        max_len : int, optional
            Maximal length of cycle to consider (default 20)
        weighting : Type, optional
            Weighting function, if None `lambda x: 1./len(x)` is taken
            (default None)
        weighting_element : Type, optional
            Weighting function per element, if None `lambda x, p: x`
            is taken (default None)
        node_entropies : 1d numpy array, optional
            Node entropies to use, if None entropies from Markov 
            transition probabilities of internal sequence are used 
            (default None)
        graph : :class:`networkx.DiGraph`, optional
            Graph to operate on, if None Markov graph belonging
            to internal sequence is used (default None)

        Returns
        -------
        (paths, scores) : (1d numpy array, 1d numpy array) 
            Scored paths, where paths is a 1d array of paths and scores a
            1d array of cycle scores. Index in scores identical to index in paths,
            arrays sorted by score (ascending).
        """
        if graph is None:
            graph = self.markov_graph
        if node_entropies is None:
            node_entropies = self.label_markov_entropies
        import networkx as nx
        paths = [p for p in nx.all_simple_paths(graph, node1, node2, cutoff=max_len) if len(p) >= min_len]

        if weighting is None:
            weighting = lambda x: 1. / len(x)
        if weighting_element is None:
            weighting_element = lambda x, p: x
        scores = np.array(
            [sum([weighting_element(node_entropies[m], i) for i, m in enumerate(path)]) * weighting(path)
             for path in paths])
        sort = np.argsort(scores)
        return np.array(paths)[sort], scores[sort]


# end of source
