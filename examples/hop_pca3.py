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

N = 1024
M = 32
p = 0.2

numshow = 100 #int(N*1.5)
S = (M + 2*M) * numshow

amin = int(N * p) #int(N * p * .9)
amax = int(N * p * 1.1)

#pats = np.array([sample_from_bernoulli([p]*N,1)] + [sample_from_bernoulli([np.random.randint(amin, amax + 1)/float(N)]*N,1) for _ in xrange(M - 1)])

pats = np.array([np.zeros(N) for _ in xrange(M)])
for i in xrange(M):
    pats[i][i*int(amin*0.1):i*int(amin*0.1) + amin] = 1

patsrc = sample_from_bernoulli([p]*M*N,1)
#pats = np.array([patsrc[i*8 : i*8 + N] for i in xrange(M)])

X = np.zeros((N, S))
actual = np.zeros(X.shape)

'''
#patpos = np.random.permutation(S)[:M*numshow]
patpos = []

for i in xrange(numshow):
	patpos.extend(range(i*M, (i+1)*M))

pos = 0
for pat in pats:
	for _ in xrange(numshow):
		noise = (np.random.random((1, N)) < p)
		patnoise = np.logical_xor(pat.astype(bool), noise).astype(int).ravel()
		X[:, patpos[pos]] = patnoise

		actual[:, patpos[pos]] = pat
		pos += 1
'''

X = (np.random.random((S, N)) < p).astype(int).T

dd = 0
jj = 0
#patpos = np.random.permutation(S)[:M * numshow]
patpos = xrange(M * numshow)
for j in patpos:
	pat = pats[jj]
	noise = (np.random.random((1, N)) < p)
	patnoise = np.logical_xor(pat.astype(bool), noise).astype(int).ravel()
	X[:, j] = patnoise
	actual[:, j] = pat
	if dd == 0:
		#up
		if jj < len(pats) - 1:
			jj += 1
		else:
			jj -= 1
			dd = 1
	else:
		#down
		if jj > 0:
			jj -= 1
		else:
			jj += 1
			dd = 0

corr = np.dot(X, X.T)
corr[np.diag_indices(X.shape[0])] = 0

cov = np.cov(X)
cov[np.diag_indices(X.shape[0])] = 0

spikes = Spikes(spikes=X)

print "learning.."
learner=Learner(spikes)

t1 = now()
learner.learn_from_spikes(spikes,window_size=1)
t2 = now()


if (learner.network(pats) == 0).all():
	print "no patterns!"

numcorrect = (learner.network(pats) == pats).sum()  # == M * N
print numcorrect, numcorrect/float(N*M)
print t2-t1

rast = spikes.rasterize()
conv_spikes = learner.network(rast[:, :500].T).T
difference = actual[:, :500] - conv_spikes

perm = np.random.permutation(N)

plt.figure()
plt.imshow(rast[:, :500], interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Raw noisy)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/rast.pdf', bbox_inches='tight', pad_inches=.02)

plt.figure()
plt.imshow(rast[perm, :500], interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Raw noisy), shuffled')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/rast_shuff.pdf', bbox_inches='tight', pad_inches=.02)


plt.figure()
plt.imshow(conv_spikes, interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Converged Patterns)')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/conv_spikes.pdf', bbox_inches='tight', pad_inches=.02)

plt.figure()
plt.imshow(conv_spikes[perm], interpolation='nearest', cmap='gray')
plt.title('First 500 time bins (Converged Patterns), shuffled')
plt.xlabel('Time bin')
plt.ylabel('Neuron #')
plt.savefig('figs/conv_spikes_shuff.pdf', bbox_inches='tight', pad_inches=.02)


# actual = np.zeros(X.shape)
# actual[:, patpos] += 1
plt.figure()
plt.imshow(actual[:, :500], interpolation='nearest', cmap='gray')
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


print "mean difference", difference.mean()

plt.figure()
plt.imshow(cov, interpolation='nearest')
plt.title('(Off-diagonal) Covariance matrix')
plt.xlabel('Neuron #')
plt.ylabel('Neuron #')
plt.colorbar()
plt.savefig('figs/Cov.pdf')

############################################
# PCA
X_mz = X.T - X.mean(axis=1)
Cov_mz = np.cov(X)
# Cmz = np.dot(mouse_cmz.T, mouse_cmz) / mouse_cmz.shape[0]
d, v = np.linalg.eig(Cov_mz)   # eigenvalues and eigenvectors (PCA components) all v[i] norm 1
idx = np.argsort(d)
v = v[:, idx]       # normalized eigenvectors
d = d[idx]
d = d[::-1]         # ordered largest first
v = v[:, ::-1]

plt.figure()
plt.plot(d)


'''
# compute projected features
diag = np.diag(1 / np.sqrt(d))
projected_features = np.dot(X_mz.T, np.dot(v, diag))
# projected_features_3d = np.column_stack((mouse_avgs[:, 0], projected_features[:, :3]))
# col = np.column_stack((mouse_avgs[:, 0], mouse_numbers))

Cov_mz_recon = np.dot(np.dot(v, np.diag(d)), v.T)

variances_captured = np.zeros(v.shape[0] + 1)

for i in xrange(Cov_mz.shape[0] + 1):
    d_3d = np.zeros(v.shape[0])
    d_3d[0:i] = d[0:i]
    
    P_3d = np.sqrt(v.shape[0]) * np.dot(np.dot(v, np.diag(np.sqrt(d_3d))), v.T)
    P = np.sqrt(v.shape[0]) * np.dot(np.dot(v, np.diag(np.sqrt(d))), v.T)
    U = np.dot(np.linalg.inv(P), X_mz.T)

    X_mz_recon = np.dot(P_3d, U).T
    variances_captured[i] = 1 - np.linalg.norm(X_mz_recon - X_mz, ord='fro') / np.linalg.norm(X_mz, ord='fro')
    print '%d: %1.4f perc variance' % (i, variances_captured[i])
'''

# demixing with ica
import random as rnd
import numpy as np
from numpy import linalg as LA
from scipy import stats
from sklearn.decomposition import FastICA

def computes_pca(activity_matrix):

    # computes correlation matrix
    correlation_matrix = np.corrcoef(activity_matrix) 
    
    # computes principal components and loadings
    eigenvalues,pcs = LA.eig(correlation_matrix)
    
    return eigenvalues,pcs,correlation_matrix

activity_matrix = X

# zscores activity matrix
z_actmatrix = stats.zscore(activity_matrix.T)
z_actmatrix = z_actmatrix.T

# computes PCA in activity matrix. Function defined below.
eigenvalues,pcs,_ = computes_pca(activity_matrix)

q = float(np.size(z_actmatrix,1)/np.size(z_actmatrix,0))

lambda_max = pow(1+np.sqrt(1/q),2)        
nassemblies = np.sum(eigenvalues>lambda_max)
print '... number of assemblies detected: ' + str(nassemblies)
    
# ica
z_actmatrix = z_actmatrix.T
ica = FastICA()
ica.n_components=nassemblies
ica.fit(z_actmatrix).transform(z_actmatrix)
assemblypatterns = ica.components_
assemblypatterns = assemblypatterns.T

aassemblypatterns = abs(assemblypatterns)
#aassemblypatterns = aassemblypatterns[np.lexsort(np.fliplr(aassemblypatterns).T)]


plt.figure()
plt.matshow(pats.T, cmap='gray')

plt.figure()
plt.matshow(aassemblypatterns, cmap='gray')
plt.colorbar()


t = 0.00025

binpats = []
numfound = 0
for i in xrange(nassemblies):
    pp = aassemblypatterns.T[i].copy()
    pp[pp>t] = 1
    pp[pp<=t] = 0
    pp = pp.astype(int)
    binpats.append(pp)
    found = -1
    for j, ppp in enumerate(pats):
        if (pp == ppp).all():
            found = j
            numfound += 1
            break    
    print i, found

binpats = np.array(binpats).T

plt.figure()
plt.matshow(binpats, cmap='gray')

