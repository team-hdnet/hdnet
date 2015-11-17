# sound_snip.py hdnet and ON/OFF applied to sound

import numpy as np
import matplotlib.pyplot as plt

from hdnet.stimulus import Stimulus
from hdnet.spikes import Spikes
from hdnet.spikes_model import SpikeModel, BernoulliHomogeneous, DichotomizedGaussian

import scipy 

import scipy.io.wavfile

wavfl = scipy.io.wavfile.read('data/02_16k.wav')
lchnl = wavfl[1][:, 0]

rate, raw = scipy.io.wavfile.read("data/02_16k.wav")
if len(raw.shape) > 1 and raw.shape[1] > 1:
    raw = raw[:, 0]

def low_pass(a, n=16):
    shape = a.shape
    a = np.vstack([a[::-1], a, a[::-1]])
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n)[shape[0]:shape[0] + shape[0]]

time_points = 500000
raw = raw[1000000:1000000 + time_points]

from scipy.ndimage.filters import gaussian_filter1d

ws = 8  # window size
lp = gaussian_filter1d(raw, ws)
hp = raw - lp

# windowed input amplitudes
amplitudes_spikes = Spikes(hp, preprocess=False)  # will convert to binary
amplitudes_window = amplitudes_spikes.to_windowed(window_size=ws, reshape=True)

# ON/OFFify
alpha = 1/2.  # twice alpha should be smallest distance between amplitudes
ON = (amplitudes_window > alpha).astype(int)
OFF = (amplitudes_window < -alpha).astype(int)
ON_OFF = np.hstack([ON, OFF])

spikes = Spikes(ON_OFF.T[:, :])
spikes_model = SpikeModel(spikes=spikes, window_size=1)
spikes_model.fit()
spikes_model.chomp()

from hdnet.stats import SequenceAnalyzer
seq = SequenceAnalyzer(spikes_model.hopfield_patterns)
print "Entropy: %1.4f" % seq.entropy()

arr = amplitudes_window - amplitudes_window.mean(axis=1).reshape(amplitudes_window.shape[0], 1)
stimulus = Stimulus(arr)
mtas = spikes_model.hopfield_patterns.mem_triggered_stim_avgs(stimulus)

c = 0
reconstruction_bin = np.zeros(raw.shape)
reconstruction_mta = np.zeros(raw.shape)

for c2, i in enumerate(xrange(0, raw.shape[0] - 1, ws)):
	oo = spikes_model.hopfield_patterns.pattern_to_binary_matrix(spikes_model.hopfield_patterns.sequence[i])
	tmp = oo[:ws] - oo[ws:] # assumes we have no 00's
	mta = mtas[spikes_model.hopfield_patterns.sequence[i]]
	reconstruction_bin[c: c + ws] = tmp.copy()
	reconstruction_mta[c: c + ws] = mta.copy()
	c += ws

scipy.io.wavfile.write('data/bin_16_alpha_%d.wav' % int(alpha), rate, reconstruction_bin)
scipy.io.wavfile.write('data/mta_16_alpha_%d.wav' % int(alpha), rate, reconstruction_mta)
