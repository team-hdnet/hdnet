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

amin = 11
amax = 15

pats = np.array([sample_from_bernoulli([np.random.randint(amin, amax + 1)/float(N)]*N,1) for _ in xrange(M)])

#pats = np.array([np.zeros(N) for _ in xrange(M)])
#for i in xrange(M):
#    pats[i][16*i:16*(i+1)] = 1

X = sample_from_bernoulli([p]*N, S)

actual = np.zeros(X.shape)

numshow = int(S * p / 10.)
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


binpats = []
t = 0.0002
for i in xrange(M):
    pp = aassemblypatterns.T[i].copy()
    pp[pp>t] = 1
    pp[pp<=t] = 0
    pp = pp.astype(int)
    binpats.append(pp)
    found = -1
    for j, ppp in enumerate(pats):
        if (pp == ppp).all():
            found = j
            break    
    print i, found


