Introduction
============

`hdnet` is a Python package for analysis of neural population spiking
data, i.e. parallel spike trains.

In particular, it provides a novel method for finding and extracting
salient low-dimensional representations of the dynamics of populations
of spiking neurons based on a denoising approach to spatiotemporal
patterns (STP) contained in the data.

Finding STP is classical problem in data analysis of parallel spike
trains, and quite a number of approaches to detect and classify
recurring spatiotemporal patterns (STP) of neural population activity
were proposed :cite:`Gruen2010`.

Yet, most published methods so far either focus solely on synchrony
detection :cite:`Pipa2008`, :cite:`PicadoMuino2013`,
:cite:`LopesDosSantos2013`
or assume a more or less noiseless scenario, seeking to classify
exactly recurring STP in neuronal activity (apart from allowing
some jitter in spike timing), see e.g. :cite:`Gansel2012`.

Given the usually high variability of population responses to stimuli,
the re-occurrence of such exactly repeating STP becomes more and more
unlikely with increasing population size, though. Assuming that despite
this variability, network activity is not random per se (under the
experimentally well-supported hypothesis :cite:`Abeles1993` :cite:`Arieli1995`
that the population has to code information about stimuli in some form
of STP), a much more plausible situation is that some underlying STP
appears in several "corrupted" variants,  both expressing jitter in
spike times and differing in a few missing or excess spikes (characterized
by a low Hamming distance to a true, underlying STP).

Our method takes a different aproach. Using Hopfield networks trained
with *minimum probability flow* (MPF), the occuring raw spatiotemporal
patterns are grouped into clusters of similar patterns in an unsupervised
way, assigning to each cluster a *memory* (the fixed point of the
Hopfield dynamics in each cluster).

The proposed method is robust to this variability in the signal and able to
extract the underlying recurring patterns, even for seldomly occurring STP
and large population sizes.

See Section :ref:`section-mathematical-background` for an introduction of
the mathematical background behind some of the techniques used in `hdnet`.
