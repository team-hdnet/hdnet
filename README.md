hdnet
=====

`hdnet` stands for Hopfield denoising network. It is a Python package
for analysis of neural population spiking data, i.e. parallel spike trains.

In particular, it provides a novel method for finding and extracting
salient low-dimensional representations of the dynamics of populations
of spiking neurons based on a denoising approach to spatiotemporal
patterns (STP) contained in the data.

Finding STP is classical problem in data analysis of parallel spike
trains, and quite a number of approaches to detect and classify
recurring spatiotemporal patterns (STP) of neural population activity
were proposed [1].

Yet, most published methods so far either focus solely on synchrony
detection [2, 3, 4] or assume a more or less noiseless scenario, seeking to
classify exactly recurring STP in neuronal activity (apart from allowing
some jitter in spike timing), see e.g. [5]. Some furthermore suffer from
effects of combinatorial explosion [3, 5], making their application to
highly parallel neural recordings a hard computational problem.

Given the usually high variability of population responses to stimuli,
the re-occurrence of such exactly repeating STP becomes more and more
unlikely with increasing population size, though. Assuming that despite
this variability, network activity is not random per se (under the
experimentally well-supported hypothesis [6, 7] that the population has
to code information about stimuli in some form of STP), a much more
plausible situation is that some underlying STP appears in several
"corrupted" variants,  both expressing jitter in spike times and differing
in a few missing or excess spikes (characterized by a low Hamming distance
to a true, underlying STP).

Our method takes a different aproach. Using Hopfield networks trained
with *minimum energy flow* (MEF), the occuring raw spatiotemporal
patterns are grouped into clusters of similar patterns in an unsupervised
way, assigning to each cluster a *memory* (the fixed point of the
Hopfield dynamics in each cluster).

The proposed method is robust to this variability in the signal and able to
extract the underlying recurring patterns, even for seldomly occurring STP
and large population sizes.

The package ships with a tutorial and is extensively documented.

References
----------
Papers Using HDNET:

[A] Hillar, Christopher and Sohl-Dickstein, Jascha and Koepsell, Kilian, Efficient and optimal binary Hopfield associative memory storage using minimum probability flow, 2012.
https://arxiv.org/abs/1204.2916v1

[B] Hillar, Christopher and Mehta, Ram and Koepsell, Kilian, A Hopfield recurrent neural network trained on natural images performs state-of-the-art image compression, 2014 IEEE International Conference on Image Processing (ICIP), 2014.

[C] Hillar, Christopher, and Felix Effenberger. "Robust discovery of temporal structure in multi-neuron recordings using hopfield networks." Procedia Computer Science 53 (2015): 365-374.
https://www.sciencedirect.com/science/article/pii/S1877050915018165

[D] Effenberger, Felix, and Christopher Hillar. "Discovery of salient low-dimensional dynamical structure in neuronal population activity using hopfield networks." In International Workshop on Similarity-Based Pattern Recognition, pp. 199-208. Springer, Cham, 2015.
https://link.springer.com/chapter/10.1007/978-3-319-24261-3_16

[E] Mehta, Ram and Marzen, Sarah and Hillar, Christopher, Exploring discrete approaches to lossy compression schemes for natural image patches, 2015 23rd European Signal Processing Conference (EUSIPCO), 2015.

[F] Hillar, Christopher and Marzen, Sarah, Revisiting perceptual distortion for natural images: Mean discrete structural similarity index, 2017 Data Compression Conference (DCC), 2017.

[G] Hillar, Christopher J and Marzen, Sarah E, Neural network coding of natural images with applications to pure mathematics, Contemporary Mathematics; American Mathematical Sociecty: Providence, RI, USA, 2017.

[H] Hillar, Christopher J and Tran, Ngoc M, Robust exponential memory in Hopfield networks, The Journal of Mathematical Neuroscience, 2018.
https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0056-2

[I] Hillar, Chan, Taubman, Rolnick, Hidden Hypergraphs, Error-Correcting Codes, and Critical Learning in Hopfield Networks, Entropy, 2021.
https://www.mdpi.com/1099-4300/23/11/1494/htm

[J] Lamberti, Martina and Hess, Michael and Dias, In{\^e}s and van Putten, Michel and le Feber, Joost and Marzen, Sarah, Maximum entropy models provide functional connectivity estimates in neural networks, 2022.

Others:

[1] S. Grün and S. Rotter. Analysis of parallel spike trains. Springer, 2010  
[2] G. Pipa, D. W. Wheeler, W. Singer, and D. Nikolić. NeuroXidence: reliable 
and efficient analysis of an excess or deficiency of joint-spike events. 
Journal of Computational Neuroscience, 25(1):64–88, 2008.  
[3] D. Picado-Muiño, C. Borgelt, D. Berger, G. Gerstein, and S. Grün. Finding
neural assemblies with frequent item set mining.
Frontiers in neuroinformatics, 7(May):9, 2013.  
[4] V. Lopes-dos-Santos, S. Ribeiro, and A. B. L. Tort. Detecting cell assemblies
in large neuronal populations. Journal of Neuroscience Methods, 220(2):149–66, 2013.  
[5] K. S. Gansel and W. Singer. Detecting multineuronal temporal patterns in 
parallel spike trains. Frontiers in Neuroinformatics, 6(May):18, 2012.  
[6] M. Abeles and H. Bergman. Spatiotemporal firing patterns in the frontal cortex 
of behaving monkeys. Journal of Neurophysiology, 1993.  
[7] A. Arieli and D. Shoham. Coherent spatiotemporal patterns of ongoing activity
revealed by real-time optical imaging coupled with single-unit recording in the cat
visual cortex. Journal of Neurophysiology, 1995.


Installing
----------

Until we provide an official release, please install hdnet directly from Github using `pip`.

To install / update to the latest version use:

`pip install git+ssh://git@github.com/team-hdnet/hdnet.git`

To install a specific version, use:

`pip install git+ssh://git@github.com/team-hdnet/hdnet@VERSION`,

where `VERSION` is `v0.1` for example. See releases tab above for released versions.
 

Documentation
-------------

[View the documentation (online and PDF version available).](http://team-hdnet.github.io/hdnet) 


Citing hdnet
------------

If you use `hdnet` in your work please cite it using the following BibTeX entry.

```
@online{hdnet,
    author      = {Hillar, Christopher  and Effenberger, Felix},
    title       = {\texttt{hdnet} -- a Python package for parallel spike train analysis},
    year        = {2015},
    url         = {https://github.com/team-hdnet/hdnet}
}
```


Authors
-------

[Christopher Hillar](http://www.msri.org/people/members/chillar/)  
Redwood Center for Theoretical Neuroscience  
UC Berkeley, Berkeley, CA, USA  
E-mail: hillarmath@gmail.com

[Felix Effenberger](http://personal-homepages.mis.mpg.de/effen/)  
Max-Planck-Institute for Mathematics in the Sciences  
Leipzig, Germany  
E-mail: Felix.Effenberger@mis.mpg.de


License
-------

For non-commercial use, `hdnet` is licensed under the GNU GPLv3, see file LICENSE.
For use in commercial projects please contact the authors for licensing options. 

Contributing
------------

We are looking for contributors! If you are interested in contributing 
or find a bug, please contact us personally or via a pull request!


Building the documentation
--------------------------

* Install required modules
  pip install sphinx sphinx_rtd_theme sphinxcontrib-napoleon sphinxcontrib-bibtex sphinxjp.themes.basicstrap

* Build documentation:
  HTML: make html
  LaTeX: make latexpdf

* Built documentation can be found in doc/_build/html and doc/_build/latex

