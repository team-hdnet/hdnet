# First examples with HDNET
#
# C. Hillar, Dec. 2014
#

import numpy as np
import matplotlib.pyplot as plt

from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, BernoulliHomogeneous
from hdnet.visualization import raster_plot_psth, pattern_rank_plot


# Let's make up some spikes
spikes_arr = (np.random.random((30, 300)) < .05).astype(int)
spikes = Spikes(spikes_arr=spikes_arr[:, :])

# let's look at them: quick save as PNG or make PSTH pyplot
spikes.rasterize(save_png_name='raster')
fig, ax_scatter, ax_hist_x, ax_hist_x, ax_hist_y = raster_plot_psth(spikes.spikes_arr, bin_size=0.002)
plt.show()

# let's examine the structure in spikes using a spike modeler
# the basic modeler trains a Hopfield network using MPF on the raw spikes
spike_model = SpikeModel(spikes=spikes)
spike_model.fit()
spike_model.chomp()





# # now let's take a sample from the model of the data
# BH_sample_spikes = spike_model.sample_from_model()
# BH_sample_spikes.rasterize(save_png_name='raster_BH_model_sample')
# print "%1.4f should equal around .05" % BH_sample_spikes.spikes_arr.mean()





# spike_model = SpikeModel(spikes=spikes)



# let's model them as DichotomizedGaussianPoisson:
# from the paper: Generating spike-trains with specified correlations, Macke et al.
# www.kyb.mpg.de/bethgegroup/code/efficientsampling
