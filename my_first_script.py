# First simulated data examples with HDNET
#
# C. Hillar, Dec. 2014
#

import numpy as np
import matplotlib.pyplot as plt

from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, BernoulliHomogeneous, DichotomizedGaussian


# Let's first make up some simuilated spikes: 2 trials
spikes_arr = (np.random.random((2, 10, 200)) < .05).astype(int)
spikes_arr[0, [1, 5], ::5] = 1  # insert correlations
spikes_arr[1, [2, 3, 6], ::7] = 1  # insert correlations
spikes = Spikes(spikes_arr=spikes_arr)

# let's look at them: quick save as PNG or make PSTH pyplot
plt.matshow(spikes.rasterize(), cmap='gray')
plt.title('Raw spikes')
#spikes.rasterize(save_png_name='raster')
plt.matshow(spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Raw spikes covariance')
#spikes.covariance(save_png_name='simul_cov_matrices')

# let's examine the structure in spikes using a spike modeler
spike_model = BernoulliHomogeneous(spikes=spikes)
BH_sample_spikes = spike_model.sample_from_model()
plt.matshow(BH_sample_spikes.rasterize(), cmap='gray')
plt.title('BernoulliHomogeneous sample')
print "%1.4f means" % BH_sample_spikes.spikes_arr.mean()
plt.matshow(BH_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('BernoulliHomogeneous covariance')

# let's model them as DichotomizedGaussian:
# from the paper: Generating spike-trains with specified correlations, Macke et al.
# www.kyb.mpg.de/bethgegroup/code/efficientsampling
spike_model = DichotomizedGaussian(spikes=spikes)
DG_sample_spikes = spike_model.sample_from_model()
plt.matshow(DG_sample_spikes.rasterize(), cmap='gray')
plt.title('DichotomizedGaussian sample')
print "%1.4f means" % DG_sample_spikes.spikes_arr.mean()
plt.matshow(DG_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('DichotomizedGaussian covariance')

# the basic modeler trains a Hopfield network using MPF on the raw spikes
spike_model = SpikeModel(spikes=spikes)
spike_model.fit()  # note: this fits a single network to all trials
spike_model.chomp()
converged_spikes = Spikes(spikes_arr=spike_model.hopfield_spikes_arr)
plt.matshow(converged_spikes.rasterize(), cmap='gray')
plt.title('Converge dynamics on Raw data')
plt.matshow(converged_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Covariance of converged memories')

# let's examine more carefully the memories discovered in the Hopnet
# plot memory label (its chronological appearance) as a function of time
plt.figure()
plt.scatter(range(len(spike_model.memories.sequence)), 1 + np.array(spike_model.memories.sequence))
plt.xlabel('time bin')
plt.ylabel('Memory number (chronological order of appearance)')
plt.title('Converged memory label at each time bin')
# versus the Raw data
plt.figure()
plt.scatter(range(len(spike_model.emperical.sequence)), 1 + np.array(spike_model.emperical.sequence))
plt.ylabel('Raw pattern number (chronological order of appearance)')
plt.xlabel('time bin')
plt.title('Raw pattern label at each time bin')

# memories are ordered by their first appearance:
bin_memories = spike_model.memories.fp_list
arr = np.zeros((spike_model.original_spikes.N, 2 * len(bin_memories)))
for c, memory in enumerate(bin_memories):
    arr[:, c] = spike_model.memories.fp_to_binary_matrix(c)
for c, memory in enumerate(bin_memories):
    arr[:, c + len(bin_memories)] = spike_model.memories.stas[memory] / spike_model.memories.counts[memory]

print "Probabilities of each memory:"
print zip(bin_memories, spike_model.memories.to_prob_vect())

