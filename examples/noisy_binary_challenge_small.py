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
S = int(1e5)
M = 16
p = 0.05

amin = 8
amax = 16

pats = np.array([sample_from_bernoulli([np.random.randint(amin, amax + 1)/float(N)]*N,1) for _ in xrange(M)])


X = sample_from_bernoulli([p]*N, S)

numshow = int(S * p / 5.)
patpos = np.random.permutation(S)[:M*numshow]
pos = 0
for pat in pats:
	for _ in xrange(numshow):
		noise = (np.random.random((1, N)) < p)
		patnoise = np.logical_xor(pat.astype(bool), noise).astype(int).ravel()
		X[:, patpos[pos]] = patnoise
		pos += 1

corr = np.dot(X, X.T)
corr[np.diag_indices(X.shape[0])] = 0

window_size = 1
spikes = Spikes(spikes=X)

print "learning.."
learner=Learner(spikes)

t1 = now()
learner.learn_from_spikes(spikes,window_size=window_size)
t2 = now()

print "converging.."
patterns=PatternsHopfield(learner)
patterns.chomp_spikes(spikes,window_size=window_size)

print (learner.network(pats) == pats).sum()  # == M * N
print t2-t1

#empirical = Counter(spikes)
#empirical.chomp_spikes(spikes,window_size=window_size)

#print 'Hop patterns: %d | Empirical: %d' % (len(patterns), len(empirical))
#found = '1'*assembly_size+'0'*(N-assembly_size) in patterns.fp_list
#print 'found', found

# J=learner.network.J
# np.savez('noisynet.npz'), J=J, C=corr, fps=patterns.fp_list, found = found)
