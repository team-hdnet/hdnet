# First simulated data examples with HDNET
#
# C. Hillar, Dec. 2014
#


import numpy as np
import matplotlib.pyplot as plt

from hdnet.stimulus import Stimulus
from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, BernoulliHomogeneous, DichotomizedGaussian

# Let's first make up some simuilated spikes: 2 trials
spikes = (np.random.random((2, 10, 200)) < .05).astype(int)
spikes[0, [1, 5], ::5] = 1  # insert correlations
spikes[1, [2, 3, 6], ::11] = 1  # insert correlations
spikes = Spikes(spikes=spikes)

# let's look at them: quick save as PNG or make PSTH pyplot
plt.matshow(spikes.rasterize(), cmap='gray')
plt.title('Raw spikes')
#spikes.rasterize(save_png_name='raster')
plt.matshow(spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Raw spikes covariance')
#spikes.covariance(save_png_name='simul_cov_matrices')

# let's examine the structure in spikes using a spike modeler
spikes_model = BernoulliHomogeneous(spikes=spikes)
BH_sample_spikes = spikes_model.sample_from_model()
plt.matshow(BH_sample_spikes.rasterize(), cmap='gray')
plt.title('BernoulliHomogeneous sample')
print "%1.4f means" % BH_sample_spikes.spikes.mean()
plt.matshow(BH_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('BernoulliHomogeneous covariance')

# let's model them as DichotomizedGaussian:
# from the paper: Generating spike-trains with specified correlations, Macke et al.
# www.kyb.mpg.de/bethgegroup/code/efficientsampling
spikes_model = DichotomizedGaussian(spikes=spikes)
DG_sample_spikes = spikes_model.sample_from_model()
plt.matshow(DG_sample_spikes.rasterize(), cmap='gray')
plt.title('DichotomizedGaussian sample')
print "%1.4f means" % DG_sample_spikes.spikes.mean()
plt.matshow(DG_sample_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('DichotomizedGaussian covariance')

# the basic modeler trains a Hopfield network using MPF on the raw spikes
spikes_model = SpikeModel(spikes=spikes)
spikes_model.fit()  # note: this fits a single network to all trials
spikes_model.chomp()
converged_spikes = spikes_model.hopfield_spikes
plt.matshow(converged_spikes.rasterize(), cmap='gray')
plt.title('Converge dynamics on Raw data')
plt.matshow(converged_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
plt.title('Covariance of converged memories')

# let's examine more carefully the memories discovered in the Hopnet
# plot memory label (its chronological appearance) as a function of time
plt.figure()
plt.scatter(range(len(spikes_model.hopfield_patterns.sequence)), 1 + np.array(spikes_model.hopfield_patterns.sequence))
plt.xlabel('time bin')
plt.ylabel('Memory number (chronological order of appearance)')
plt.title('Converged memory label at each time bin')
# versus the Raw data
plt.figure()
plt.scatter(range(len(spikes_model.raw_patterns.sequence)), 1 + np.array(spikes_model.raw_patterns.sequence))
plt.ylabel('Raw pattern number (chronological order of appearance)')
plt.xlabel('time bin')
plt.title('Raw pattern label at each time bin')

# memories are ordered by their first appearance:
bin_memories = spikes_model.hopfield_patterns.patterns
arr = np.zeros((spikes_model.original_spikes.N, 2 * len(bin_memories)))
for c, memory in enumerate(bin_memories):
    arr[:, c] = spikes_model.hopfield_patterns.pattern_to_binary_matrix(c)
for c, memory in enumerate(bin_memories):
    arr[:, c + len(bin_memories)] = spikes_model.hopfield_patterns.mtas[memory] / spikes_model.hopfield_patterns.counts[memory]

# print "Probabilities of each memory:"
# print zip(bin_memories, spikes_model.hopfield_patterns.label_probabilities())

# Saving / Loading
spikes_model.save('my_spikes_model')
spikes_model = SpikeModel.load('my_spikes_model')




# (Fake) Stimuli
calvin = np.load('examples/data/calvin.npy')  # 90 by 100 numpy array
hobbes = np.load('examples/data/hobbes.npy')

stimulus_arr = 20 * np.random.randn(2, 200, *calvin.shape)
stimulus_arr[0, ::5] = calvin + 50 * np.random.randn(200 / 5, *calvin.shape)
stimulus_arr[1, ::11] = hobbes + 50 * np.random.randn(200 / 11 + 1, *hobbes.shape)

plt.matshow(stimulus_arr[0, 0], cmap='gray')
plt.title('Calvin Sample Stimulus')
plt.matshow(stimulus_arr[1, 0], cmap='gray')
plt.title('Hobbes Sample Stimulus')

stimulus = Stimulus(stimulus_arr=stimulus_arr)
avgs = spikes_model.hopfield_patterns.mem_triggered_stim_avgs(stimulus)

for stm_avg in avgs:
    plt.matshow(stm_avg, cmap='gray')
    plt.title('Memory Triggered Stimulus Average')



# Real Data
# spikes = Spikes(spk_folder='data/Blanche/crcns_pvc3_cat_recordings/drifting_bar/spike_data')
# spikes_model = SpikeModel(spikes=spikes)
# spikes_model.fit()  # note: this fits a single network to all trials
# spikes_model.chomp()
# converged_spikes = Spikes(spikes_arr=spikes_model.hopfield_spikes)
# from hdnet.data import SpkReader
# fn = 'data/Blanche/crcns_pvc3_cat_recordings/drifting_bar/spike_data'
# spikes = SpkReader.read_spk_folder(fn)
# spikes_model = SpikeModel(spikes=spikes)
# spikes_model.fit()  # note: this fits a single network to all trials
# spikes_model.chomp()
# converged_spikes = Spikes(spikes=spikes_model.hopfield_spikes)
# plt.matshow(converged_spikes.rasterize(), cmap='gray')
# plt.title('Converge dynamics on Raw data')
# plt.matshow(converged_spikes.covariance().reshape((2 * 10, 10)), cmap='gray')
# plt.title('Covariance of converged memories')

