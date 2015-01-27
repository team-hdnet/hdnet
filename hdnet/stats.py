# -*- coding: utf-8 -*-
"""
    hdnet.TITLE
    ~~~~~~~~~~~

    DESCRIPTION

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

__all__ = ()

import numpy as np


class Stats(object):
    def __init__(self):
        object.__init__(self)


def compute_label_probabilities(sequence):
    """
    Computes probabilities of labels as empirical probabilities over sequence
    :param sequence: 1d list or array of labels
    :return: 1d numpy array of empirical probabilities
    """
    import collections
    c = collections.Counter(sequence)
    occurrences = np.array(c.values())
    prob = occurrences.astype(float) / occurrences.sum()
    return prob


def compute_label_markov_probabilities(sequence):
    """
    Computes Markov transition probabilities of all labels occuring in sequence
    over this sequence
    :param sequence: 1d list or numpy array of labels
    :return: Markov transition probability matrix of labels in sequence as
             2d numpy array
    """
    sequence = np.atleast_1d(sequence)
    n_fp = len(set(sequence))
    m_prob = np.zeros((n_fp, n_fp), dtype=float)

    for p in xrange(n_fp):
        for pidx in np.where(sequence == p)[0]:
            if pidx == len(sequence) - 1:
                continue
            m_prob[p, sequence[pidx + 1]] += 1
        m_prob[p, :] /= m_prob[p, :].sum()

    return m_prob


def compute_label_entropy_markov(markov_probabilities, eps=1e-9):
    """
    Computes the entropy of each label using its Markov transition probabilities
    in the space of all labels
    :param markov_probabilities: Markov transition probabilities of labels
    :return: entropies of labels as 1d numpy array
    """
    h = lambda a: -sum(a * np.log2(a))
    return np.abs(np.array([h(x[x > eps]) for x in markov_probabilities.copy()]))


def remove_repeating_labels_from_sequence(sequence, repetitions=2):
    """
    Removes all consecutive repetitions occuring more than repetitions times
    (default: 2) from sequence
    :param sequence:
    :param repetitions:
    :return:
    """
    seq_filtered = []
    for i in xrange(repetitions, len(sequence)):
        if all([x == sequence[i] for x in sequence[i - repetitions:i]]):
            continue
        seq_filtered.append(sequence[i])
    return np.array(seq_filtered)


def compute_markov_graph(markov_probabilities, node_labels=None,
                         thres=0, no_loop=False):
    """
    Computes the directed state transition graph over all labels using the
    package NetworkX. Each directed edge (x, y) is assigned as weight the
    Markov transition probability from label x to label y.
    :param markov_probabilities: Markov transition probabilities of labels,
     2d numpy array
    :param node_labels: remapped labels (optional, default None)
    :param thres: weight threshold for edges; only edges above that weight
     are included in the graph (default 0)
    :param no_loop: boolean flag specifiying handling of self-loops; if set
     to True, self-loops are discarded.
    :return: NetworkX DiGraph with lables as nodes and Markov transition
     probabilities as edges
    """
    import networkx as nx

    # nodes
    n = len(markov_probabilities)
    if node_labels is None:
        node_labels = range(n)

    # edges
    edges = []
    for i in xrange(n):
        for j in xrange(n):
            if i == j and no_loop:
                continue
            if markov_probabilities[i][j] > thres:
                k = node_labels[i]
                l = node_labels[j]
                edges.append((k, l, {'weight': markov_probabilities[k][l]}))

    # construct graph
    transition_graph = nx.DiGraph()
    transition_graph.add_nodes_from(node_labels)
    transition_graph.add_edges_from(edges)

    return transition_graph


def reduce_graph_self_loops(g):
    removed = 0
    for n in g.nodes():
        if g.has_edge(n, n):
            g.remove_edge(n, n)
            removed += 1
    return removed

def reduce_graph_brute(g, filtered_nodes):
    # remove all nodes not in filtered_nodes
    for n in g.nodes():
        if not n in filtered_nodes:
            g.remove_node(n)
    return g


def reduce_graph_bridge(g):
    # reduce bridge at v: u -> v -> w
    # becomes u -> w
    removed = 1
    while removed > 0:
        removed = 0
        for n in g.nodes():
            pred = g.predecessors(n)
            succ = g.successors(n)
            if len(pred) == 1 and len(succ) == 1:
                print "removing", succ[0]
                weight = float(np.mean([g.get_edge_data(pred[0],n)['weight'], g.get_edge_data(n,succ[0])['weight']]))
                g.add_edge(pred[0], succ[0], weight=weight)
                g.remove_node(n)
                removed += 1
    return g


def reduce_graph_stub(g):
    # recursively remove all stub nodes with no successors 
    removed = 1
    while removed > 0:
        removed = 0
        for n in g.nodes():
            if len(g.predecessors(n)) + len(g.successors(n)) <= 1:
                    g.remove_node(n)
                    removed += 1
    return g


def reduce_graph_loop(g):
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


def reduce_graph_nloop(g, n, node):
    import networkx as nx

    pred = g.predecessors(node)
    for p in pred:
        for path in nx.all_simple_paths(g, node, p):
            if len(path) >= n or not g.has_edge(path[-1], node):
                continue
            g.remove_edge(path[-1], node)


def reduce_graph_triangles(g):
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


def reduce_graph_out_degree(g, thres):
    removed = 0
    for n in g.nodes():
        if g.out_degree(n) > thres:
            g.remove_node(n)
            removed += 1
    return removed


def calculate_loops_entropy_scores(g, n, node_entropies, min_len=5, max_len=20, weighting=None, weighting_element=None):
    """
    Calculate entropy scores of loops (simple closed paths) in g starting and
    terminating in given node n. An entropy score of a path is the sum of the
    entropies (as specified in node_entropies) of each node contained in the path.
    :param g: NetworkX graph
    :param n: base node
    :param node_entropies: 1d array of node entropies
    :return: (loops, scores), where loops is a 1d array of loops and scores a
     1d array of loop scores. Index in scores identical to index in loops,
     arrays sorted by score (ascending).
    """
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


def calculate_paths_entropy_scores(g, n1, n2, node_entropies, min_len=5, max_len=20):
    """
    Calculate entropy scores of all simple paths in g from n1 to n2.
    An entropy score of a path is the sum of the entropies (as specified in
    node_entropies) of each node contained in the path.
    :param g: NetworkX graph
    :param n1: start node
    :param n2: end node
    :param node_entropies: 1d array of node entropies
    :return: (paths, scores), where paths is a 1d array of paths and scores a
     1d array of path scores. Index in scores identical to index in paths,
     arrays sorted by score (ascending).
    """
    import networkx as nx
    paths = [ p for p in nx.all_simple_paths(g, n1, n2, cutoff=max_len) if len(p) >= min_len]
    scores = np.array([sum([node_entropies[m] for m in path]) / 1. * len(path) for path in paths])
    sort = np.argsort(scores)

    return np.array(paths)[sort], scores[sort]

