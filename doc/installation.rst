Installing hdnet
================

The simplest way to install HDNet is to first clone the HDNet repo to your own
computer and then run setup.py.

* Install some required packages

    .. code::

        git clone https://github.com/team-hdnet/hdnet.git

	Then get into the directory, where you've cloned HDNet and run:
	
    .. code::
    	 
    	 python setup.py
    	 
* A Simpler way which may or may not work for many would be using 'pip' :
	
    .. code::
    	
    	pip install git+ssh://git@github.com/team-hdnet/hdnet.git
    	
    	and if you want a specific version then better use:
    	
    .. code::
    	
    	pip install git+ssh://git@github.com/team-hdnet/hdnet@VERSION
    	
    	where 'VERSION' is v0.1 for example.
    	
    	
A few more dependencies: NetworkX and PyGraphViz  	
-------------------------------------------------

NetworkX is the Python Library for dealing with Graphs and Networks, which we'll be
using a lot in HDNet, PyGraphViz is used for the visualization requirements of 
NetworkX, when we want to draw Markox Models in order to gain better intuition.

* Install NetworkX using pip

	.. code::
		
		pip install networkx

* Install PyGraphViz: This would first require getting graphviz and a C/C++ compiler
like G++(and of course Python 3.6 or higher!):

* Install GraphViz by using the following commands according to your OS:

For Ubuntu and Debian:

	.. code::
		
		sudo apt-get install graphviz graphviz-dev

For Fedora and Red Hat:

	.. code::
		
		sudo dnf install graphviz graphviz-devel
		
For MacOS(using Homebrew package manager):

	.. code ::
		
		brew install graphviz
		
* Once, you're done with that install PyGraphViz using 'pip':

	.. code ::
		
		pip install pygraphviz
		
Adding OpenBLAS support
-----------------------

The training process of Hopfield networks (from the classes :class:`.HopfieldNet`
and :class:`.HopfieldNetMPF`) can be easily parallelized to exploit all available
CPU cores by using a `numpy` installation that makes use of a multi-threaded
linear algebra library such as `OpenBLAS`.

For many Linux installations though, the default linear algebra library used by
`numpy` is not multithreaded.

For Debian based systems (such as Ubuntu), to compile `numpy` with support for
OpenBLAS, a multithreaded linear algebra library, do the following (as described
in `the following Stackoverflow topic <http://stackoverflow.com/a/14391693>`_):

* Install some required packages

    .. code::

        apt-get install build-essential gfortran python-dev git

* Create a temporary directory to compile the code

    .. code::

        mkdir /tmp/openblas && cd /tmp/openblas

* Get and compile the OpenBLAS sources. Change the install path from `/opt/OpenBLAS`
  in the example to your liking, also something like `~/.local/openblas` is possible.

    .. code::

        git clone git://github.com/xianyi/OpenBLAS
        cd OpenBLAS && make FC=gfortran
        sudo make PREFIX=/opt/OpenBLAS install
        sudo ldconfig

* Get the `numpy` sources

    .. code::

        git clone https://github.com/numpy/numpy
        cd numpy

* Adjust the build properties of numpy (to make it use the OpenBLAS library)

    .. code::

        cp site.cfg.example site.cfg

    Now open your favorite editor (`vim`, is it? Or `emacs`?) and uncomment the
    following lines:

    .. code::

        ...
        [openblas]
        libraries = openblas
        library_dirs = /opt/OpenBLAS/lib
        include_dirs = /opt/OpenBLAS/include
        ...

* Compile numpy with OpenBLAS support (optionally add the `--user` flag to the last
  python call to install the package only for the current user not using root rights;
  the optional argument is marked with parentheses [] below)
  (NOTE: The first command may not work for the latest versions of OpenBLAS and hence, it can be skipped!)

    .. code::

        python setup.py config
        python setup.py build && python setup.py [--user] install

* Now you can test performance with the following script

    .. code::

        import numpy
	from numpy.distutils.system_info import get_info
	import sys
	import timeit

	print("version: %s" % numpy.__version__)
	print("maxint:  %i\n" % sys.maxsize)

	info = get_info('blas_opt')
	print('BLAS info:')
	for kk, vv in info.items():
    	print(' * ' + kk + ' ' + str(vv))

	setup = "import numpy; x = numpy.random.random((1000, 1000))"
	count = 10

	t = timeit.Timer("numpy.dot(x, x.T)", setup=setup)
	print("\ndot: %f sec" % (t.timeit(count) / count))

    Save it as `dot_performance.py` and run the following, where `X` is the number of
    CPU cores `numpy` should use for linear algebra operations:

    .. code::

        OMP_NUM_THREADS=X python dot_performance.py

    You should see a nice speedup for higher values of `X`.
