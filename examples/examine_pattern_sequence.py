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
# non-vanishing conditional probability of occurrence in the pattern sequence.
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
#   cycles (closed paths) starting and ending at such a central node can give
# insight on how the network is driven out of its resting state (often by some
# stimulus) and enters a transient excited state before falling back to the
# resting state.
#   The code below enumerates such cycles (if existent) and sorts them by their
# (scored) entropy, a proxy measure for how reliably the network dynamics visit
# those cycles (i.e. excited states) in the data considered.

import numpy as np
import matplotlib as mpl
#to set mpl backend:
#mpl.use('Agg')
import matplotlib.pyplot as plt

from hdnet.patterns import PatternsHopfield
from hdnet.stats import SequenceAnalyzer
from hdnet.visualization import combine_windows, plot_graph

# load coverged patterns here

n = NUMBER_OF_NEURONS
ws = WINDOW_SIZE
pattern_file = 'SAVED_PATTERNS_FILE'

# load pattern sequence
patterns = PatternsHopfield.load(pattern_file)
sequence = patterns.sequence
labels = set(sequence)
n_labels = len(labels)

# create sequence analyzer instance
sa = SequenceAnalyzer(patterns)

#optionally filter labels to remove occurrences of repeated labels
FILTER_LABEL_SEQUENCE = False
if FILTER_LABEL_SEQUENCE:
    # NB. this alters label probabilities and Markov transition probabilities
    sa.filter_sequence_repeating_labels(repetitions=2)

# compute probabilities of labels, markov transition probabilities and
label_probabilities = sa.compute_label_probabilities()
markov_probabilities = sa.compute_label_markov_probabilities()
label_entropy = sa.compute_label_markov_entropies()

# plot label probabilities, markov transition probabilities and node entropy
fig, ax = plt.subplots()
ax.hist(label_probabilities, weights=[1. / n_labels] * n_labels,
         range=(label_probabilities.min(), 0.005),
         bins=100, color='k')
ax.set_xlabel('probability')
ax.set_ylabel('fraction')
ax.set_yscale('log', nonposy='clip')
ax.set_xscale('log', nonposx='clip')
plt.tight_layout()
plt.savefig('label_probabilities.png')
plt.savefig('label_probabilities.pdf')
plt.close()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
cmap = mpl.cm.autumn
cmap.set_bad('k')
mp_masked = np.ma.masked_where(markov_probabilities < 0.001 , markov_probabilities)
im = ax.matshow(mp_masked, cmap=cmap,
            norm=mpl.colors.LogNorm(vmin=0.001, vmax=1))

ax.set_xlabel('to pattern')
ax.set_ylabel('from pattern')
ax.xaxis.set_ticks([0, 500])
ax.yaxis.set_ticks([0, 500])
plt.colorbar(im)
plt.savefig('label_probabilities_markov.png')
plt.savefig('label_probabilities_markov.pdf')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots()
plt.hist(label_entropy,
         weights=[1. / n_labels] * n_labels, bins=50, color='k')
plt.xlabel('entropy')
plt.ylabel('fraction')
plt.yscale('log', nonposy='clip')
plt.tight_layout()
plt.savefig('label_entropy.png')
plt.savefig('label_entropy.pdf')
plt.close()

# construct markov graph
markov_graph = sa.compute_markov_graph()
print "Markov graph has %d nodes, %d edges" % (len(markov_graph.nodes()),
                                               len(markov_graph.edges()))

# reduce markov graph to most likely occurring labels
# adjust threshold if needed
threshold = 20
sa.reduce_graph_brute(np.argsort(label_probabilities)[::-1][:threshold])

# plot markov graph
plot_graph(markov_graph)
plt.locator_params(nbins=3)
plt.savefig('markov_graph_filtered.png')

# plot memory triggered averages for all nodes of markov graph
fig, ax = plt.subplots(threshold / 10, 10) 
for i, node in enumerate(markov_graph.nodes()):
    ax = plt.subplot(threshold / 10, 10, i + 1)
    ax.matshow(patterns.pattern_to_mta_matrix(node).reshape(n, ws),
                vmin=0, vmax=1, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('mtas.png')
plt.savefig('mtas.pdf')
plt.close()

print "filtering markov graph"
sa.reduce_graph_self_cycles()
sa.reduce_graph_triangles()
sa.reduce_graph_stub()
print "Filtered Markov graph has %d nodes, %d edges" % \
      (len(markov_graph.nodes()), len(markov_graph.edges()))

# try to guess base node (resting state memory) as node with highest degree
# (converging and diverging connections)
# -- adjust post hoc if necessary!
markov_degrees = markov_graph.degree()
base_node = max(markov_degrees, key=markov_degrees.get)
print "base node is %d" % base_node

# calculate cycles of entropies around base node
# adjust weighting and weighting per element if needed
print "calculating cycles around base node.."
weighting = lambda x: 1. / len(x)
weighting_element = lambda x, p: x / ((p + 1) * 2.) # prefer longer sequences
cycles, scores = sa.calculate_cycles_entropy_scores(
                                               base_node,
                                               min_len=3,
                                               max_len=20,
                                               weighting=weighting,
                                               weighting_element=weighting_element)
print "%d cycles" % (len(cycles))

# plot cycle statistics
n_cycles = len(cycles)
cycle_len = np.array(map(len, cycles))
fig, ax = plt.subplots() 
ax.hist(cycle_len, weights=[1. / n_cycles] * n_cycles, bins=50, color='k')
ax.set_xlabel('cycle length')
ax.set_ylabel('fraction')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('cycle_lengths.png')
plt.savefig('cycle_lengths.pdf')
plt.close()

fig, ax = plt.subplots() 
plt.hist(scores, weights=[1. / n_cycles] * n_cycles, bins=50, color='k')
plt.xlabel('cycle score')
plt.ylabel('fraction')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('cycle_scores.png')
plt.close()

fig, ax = plt.subplots() 
plt.scatter(cycle_len, scores, color='k')
plt.xlabel('cycle length')
plt.ylabel('cycle score')
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('cycle_lengths_vs_scores_scatter.png')
plt.close()

fig, ax = plt.subplots() 
plt.hist2d(cycle_len, scores, bins=100)
plt.xlabel('cycle length')
plt.ylabel('cycle score')
plt.colorbar()
plt.locator_params(nbins=3)
plt.tight_layout()
plt.savefig('cycle_lengths_vs_scores_hist.png')
plt.close()

# plot max_plot extracted cycles
# adjust if needed
max_plot = 100
interesting = np.arange(min(n_cycles, max_plot))
print "plotting averaged sequences of %d cycles.." % (len(interesting))

for i, idx in enumerate(interesting):
    cycle = cycles[idx]
    mta_sequence = [patterns.pattern_to_mta_matrix(l).reshape(n, ws)
                    for l in cycle]
    combined = combine_windows(np.array(mta_sequence))
    fig, ax = plt.subplots() 
    plt.matshow(combined, cmap='gray')
    plt.axis('off')
    plt.title('cycle %d\nlength %d\nscore %f' % \
              (idx, len(cycle), scores[idx]), loc='left')
    plt.savefig('likely-%04d.png' % i)
    plt.close()

# end of script
