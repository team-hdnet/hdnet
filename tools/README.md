hdnet Tools
===========

Each sub-directory contains one tool, described below in more detail.


matlab
------

Matlab wrapper for hdnet. Function `hdnet_fit_ising.m` that allows to fit an Ising model to given data.

Usage: Add folder to Matlab path. Call function `hdnet_fit_ising(X)` for matrix `X`,
neurons indexed by rows, samples by columns.

Needs correctly setup Matlab-Python bride, with `hdnet` installed in Python, see [Matlab help page](mathworks.com/help/matlab/ref/pyversion.html). 


vagrant
-------

A `Vagrantfile` vagrant configuration that allows for instant usage of hdnet
without prior installation. Needed tools: vagrant, virtualbox

[Vagrant](https://www.vagrantup.com/downloads.html)

[Virtualbox](https://www.virtualbox.org/wiki/Downloads)

Usage: Copy `Vagrantfile` to empty target directory. Run `vagrant up`. After
virtual machine startup open `http://localhost:8888` in browser of host machine.
Create a new Python 2 notebook. `hdnet` can be imported and used.

