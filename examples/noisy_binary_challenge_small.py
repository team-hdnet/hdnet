import numpy as np
from time import time as now
import matplotlib.pyplot as plt
import scipy.io
import os
import sys

from hdnet.spikes import Spikes
from hdnet.learner import Learner
from hdnet.patterns import PatternsHopfield
from hdnet.sampling import sample_from_bernoulli


N = 256
M = 128

numshow = 1000
S = (M + 10) * numshow

p = 0.05

amin = 11
amax = 15

pats = np.array([sample_from_bernoulli([np.random.randint(amin, amax + 1)/float(N)]*N,1) for _ in xrange(M)])


X = sample_from_bernoulli([p]*N, S)

actual = np.zeros(X.shape)

print S, numshow*M, (numshow*M) / float(S)

patpos = np.random.permutation(S)[:M*numshow]
pos = 0
for pat in pats:
	for _ in xrange(numshow):
		noise = (np.random.random((1, N)) < p)
		patnoise = np.logical_xor(pat.astype(bool), noise).astype(int).ravel()
		X[:, patpos[pos]] = patnoise

		actual[:, patpos[pos]] = pat
		pos += 1

corr = np.dot(X, X.T)
corr[np.diag_indices(X.shape[0])] = 0

cov = np.cov(X)
cov[np.diag_indices(X.shape[0])] = 0

plt.figure()
plt.imshow(cov, interpolation='nearest')
plt.title('(Off-diagonal) Covariance matrix')
plt.xlabel('Neuron #')
plt.ylabel('Neuron #')
plt.colorbar()
plt.savefig('figs/Cov.pdf')

window_size = 1
spikes = Spikes(spikes=X)

print "learning.."
learner=Learner(spikes)

t1 = now()
learner.learn_from_spikes(spikes,window_size=window_size)
t2 = now()

#print "converging.."
#patterns=PatternsHopfield(learner)
#patterns.chomp_spikes(spikes,window_size=window_size)

numcorrect = (learner.network(pats) == pats).sum()  # == M * N
print numcorrect, numcorrect/float(N*M)
print t2-t1

rast = spikes.rasterize()
conv_spikes = learner.network(rast[:, 1000:1500].T).T
difference = actual[:, 1000:1500] - conv_spikes

plt.figure()
plt.imshow(rast[:, 1000:1500], interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Raw noisy)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/rast.pdf', bbox_inches='tight', pad_inches=.02)

plt.figure()
plt.imshow(conv_spikes, interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Converged Patterns)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/conv_spikes.pdf', bbox_inches='tight', pad_inches=.02)

# actual = np.zeros(X.shape)
# actual[:, patpos] += 1
plt.figure()
plt.imshow(actual[:, 1000:1500], interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Actual hidden)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/actual_hidden.pdf', bbox_inches='tight', pad_inches=.02)

plt.figure()
plt.imshow(difference, interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Actual - Converged)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/difference_actual_found.pdf', bbox_inches='tight', pad_inches=.02)







#empirical = Counter(spikes)
#empirical.chomp_spikes(spikes,window_size=window_size)

#print 'Hop patterns: %d | Empirical: %d' % (len(patterns), len(empirical))
#found = '1'*assembly_size+'0'*(N-assembly_size) in patterns.fp_list
#print 'found', found

# J=learner.network.J
# np.savez('noisynet.npz'), J=J, C=corr, fps=patterns.fp_list, found = found)
