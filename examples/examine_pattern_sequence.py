# Analyzing neural spiking data using Hopfield networks
# - Markov probability of window labels approach
# Felix Effenberger, Jan 2015

# This assumes that a Hopfield network has already been fitted to windowed
# spike train data and the denoised patterns have been computed and saved
# via the Patterns class (see file my_first_script.py).
#   In this script basic analysis of likely occurring pattern sequences is
# carried out, based on a Markov approach.
#   The so called pattern sequence is an ordered list of memory patterns
# (fixed points of Hopfield dynamics) obtained from converging the Hopfield
# dynamics on windows of the raw spiking data. The occurring patterns are
# labeled by integer numbers (starting from 0), where each pattern is
# assigned a new label of increasing value if it has not been encountered
# before. A label thus establishes pattern identity and the pattern sequence
# consists of N labels with value 0 <= k <= N (with usually k << N).
#   After loading the denoised patterns and computing label probabilities
# and entropies of the one-step transition probabilities for each label
# (this can be thought of as a measure of how 'stably' a given sequence
# occurs in the data), a graph is constructed with the labels as nodes.
#   Edges are inserted between nodes, the labels of which have a
# non-vanishing conditional probaility of occurrence in the pattern sequence.
# The edge weight is set to this (Markov) probability.
#   In most cases, one or more 'central' node can be identified in the Markov
# graph that has/have a high in-degree (i.e. number of incoming edges),
# possibly also accompanied by a high out-degree.
#   This is characteristic for a situation in which such nodes (i.e. label,
# i.e. pattern) are the termination point (resp. starting point) of prominently
# occurring sub-sequences of patterns occurring in the pattern sequence.
#   Such a node often corresponds to some resting state of the network, that
# it repeatedly returns to. Its fixed point memory and memory triggered average
# will likely show a silent (or low activity) state of the network.
#   Loops (closed paths) starting and ending at such a central node can give
# insight on how the network is driven out of its resting state (often by some
# stimulus) and enters a transient excited state before falling back to the
# resting state.
#   The code below enumerates such loops (if existent) and sorts them by their
# (scored) entropy, a proxy measure for how reliably the network dynamics visit
# those loops (i.e. excited states) in the data considered.

import numpy as np
import matplotlib as mpl
#to set mpl backend:
#mpl.use('Agg')
import matplotlib.pyplot as plt

from hdnet.patterns import PatternsHopfield

from hdnet.stats import compute_label_probabilities, \
    compute_label_markov_probabilities, \
    compute_label_entropy_markov, compute_markov_graph, \
    reduce_graph_brute, calculate_loops_entropy_scores, \
    remove_repeating_labels_from_sequence, \
    reduce_graph_self_loops, reduce_graph_triangles, reduce_graph_stub

from hdnet.visualization import combine_windows, plot_graph


# load coverged patterns here
pattern_file = 'XXX'
n = NUMBER_OF_NEURONS
ws = WINDOW_SIZE

# load pattern sequence
patterns = PatternsHopfield.load(pattern_file)
sequence = patterns.sequence
labels = set(sequence)
n_labels = len(labels)

#optionally filter labels to remove occurrences of repeated labels
FILTER_LABEL_SEQUENCE = False
if FILTER_LABEL_SEQUENCE:
    # NB. this alters label probabilities and Markov transition probabilities
    sequence = \
        remove_repeating_labels_from_sequence(sequence, repetitions=2)

# compute probabilities of labels, markov transition probabilities and
label_probabilities = compute_label_probabilities(sequence)
markov_probabilities = compute_label_markov_probabilities(sequence)
label_entropy = compute_label_entropy_markov(markov_probabilities)

# plot label probabilities, markov transition probabilities and node entropy
fig, ax = plt.subplots()
ax.hist(label_probabilities, weights=[1. / n_labels] * n_labels,
         bins=50, color='k')
ax.set_xlabel('label')
ax.set_ylabel('fraction')
plt.yscale('log', nonposy='clip')
plt.savefig('label_probabilities.png')
plt.close()

fig, ax = plt.subplots()
mapable = ax.matshow(markov_probabilities, cmap='Blues',
            norm=mpl.colors.LogNorm(vmin=0.01, vmax=1))
ax.set_xlabel('to pattern')
ax.set_ylabel('from pattern')
plt.colorbar(mapable, ax=ax)
plt.savefig('label_probabilities_markov.png')
plt.close()

fig, ax = plt.subplots()
ax.hist(label_entropy,
         weights=[1. / n_labels] * n_labels, bins=50, color='k')
ax.set_xlabel('entropy')
ax.set_ylabel('fraction')
plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.savefig('label_entropy.png')
plt.close()

# construct markov graph
markov_graph = compute_markov_graph(markov_probabilities)
print "Markov graph has %d nodes, %d edges" % (len(markov_graph.nodes()),
                                               len(markov_graph.edges()))

# reduce markov graph to most likely occurring labels
# adjust threshold if needed
threshold = 50
reduce_graph_brute(markov_graph,
                   np.argsort(label_probabilities)[::-1][:threshold])
reduce_graph_self_loops(markov_graph)
reduce_graph_triangles(markov_graph)
reduce_graph_stub(markov_graph)
print "Filtered Markov graph has %d nodes, %d edges" % \
      (len(markov_graph.nodes()), len(markov_graph.edges()))

# plot markov graph
plot_graph(markov_graph)
plt.locator_params(nbins=3)
plt.savefig('markov_graph_filtered.png')

# plot memory triggered averages for all nodes of markov graph
for i, node in enumerate(markov_graph.nodes()):
    fig, ax = plt.subplots() 
    ax.matshow(patterns.pattern_to_mta_matrix(node).reshape(n, ws),
                vmin=0, vmax=1, cmap='gray')
    ax.set_title('node %d\nprobability %f\nentropy %f' % \
              (node, label_probabilities[node], label_entropy[node]),
              loc='left')
    plt.axis('off')
    plt.savefig('mta-%03d.png' % i)
    plt.close()


# try to guess base node (resting state memory) as node with highest degree
# (converging and diverging connections)
# -- adjust post hoc if necessary!
markov_degrees = markov_graph.degree()
base_node = max(markov_degrees, key=markov_degrees.get)
print "base node is %d" % base_node

# calculate loops of entropies around base node
# adjust weighting and weighting per element if needed
print "calculating loops around base node.."
weighting = lambda x: 1. / len(x)
weighting_element = lambda x, p: x / ((p + 1) * 2.) # prefer longer sequences
loops, scores = calculate_loops_entropy_scores(markov_graph,
                                               base_node,
                                               label_entropy,
                                               min_len=3,
                                               max_len=20,
                                               weighting=weighting,
                                               weighting_element=weighting_element)
print "%d loops" % (len(loops))

# plot loop statistics
n_loops = len(loops)
loop_len = np.array(map(len, loops))
fig, ax = plt.subplots() 
ax.hist(loop_len, weights=[1. / n_loops] * n_loops, bins=50, color='k')
ax.set_xlabel('loop length')
ax.set_ylabel('fraction')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('loop_lengths.png')
plt.close()

fig, ax = plt.subplots() 
ax.hist(scores, weights=[1. / n_loops] * n_loops, bins=50, color='k')
ax.set_xlabel('loop score')
ax.set_ylabel('fraction')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('loop_scores.png')
plt.close()

fig, ax = plt.subplots() 
ax.scatter(loop_len, scores, color='k')
ax.set_xlabel('loop length')
ax.set_ylabel('loop score')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('loop_lengths_vs_scores_scatter.png')
plt.close()

fig, ax = plt.subplots() 
mapable = ax.hist2d(loop_len, scores, bins=100)[3]
ax.set_xlabel('loop length')
ax.set_ylabel('loop score')
plt.colorbar(mapable, ax=ax)
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('loop_lengths_vs_scores_hist.png')
plt.close()

# filter out sub-loops (prefer longer ones)
filtered_idxs = []
loop_lens = np.array(map(len, loops))
sort_idxs_lens = np.argsort(loop_lens)[::-1]
for idx in sort_idxs_lens:
    loop = loops[idx]
    if any([set(loop) < set(loops[i]) for i in filtered_idxs]):
        continue
    filtered_idxs.append(idx)

filtered_idxs = np.array(filtered_idxs)
filtered_scores = scores[filtered_idxs]
sort_lens_scores = np.argsort(filtered_scores)

print "%d loops filtered" % len(filtered_idxs)

# plot max_plot extracted loops
# adjust if needed
max_plot = 100
interesting = np.arange(min(len(filtered_idxs), max_plot))

print "plotting averaged sequences of %d loops.." % (len(interesting))

for i, idx in enumerate(sort_lens_scores):
    loop = loops[filtered_idxs[idx]]
    mta_sequence = [patterns.pattern_to_mta_matrix(l).reshape(n, ws) for l in loop]
    combined = combine_windows(np.array(mta_sequence))
    fig, ax = plt.subplots() 
    ax.matshow(combined, cmap='gray')
    ax.set_title('loop %d\nlength %d\nscore %f' % \
              (idx, len(loop), scores[idx]), loc='left')
    plt.axis('off')
    plt.savefig('likely-%04d.png' % i)
    plt.close()

# end of script
