Installing hdnet
================

TBD


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

    .. code::

        python setup.py config
        python setup.py build && python setup.py [--user] install

* Now you can test performance with the following script

    .. code::

        import numpy as np
        import sys
        import timeit

        try:
            import numpy.core._dotblas
            print 'fast BLAS'
        except ImportError:
            print 'slow BLAS'

        print 'version:', np.__version__
        print 'maxint:', sys.maxint
        print

        x = np.random.random((1000,1000))

        setup = 'import numpy as np; x = np.random.random((1000,1000))'
        count = 10

        t = timeit.Timer('np.dot(x, x.T)', setup=setup)
        print 'dot:', t.timeit(count)/count, 'sec'

    Save it as `dot_performance.py` and run the following, where `X` is the number of
    CPU cores `numpy` should use for linear algebra operations:

    .. code::

        OMP_NUM_THREADS=X python dot_performance.py

    You should see a nice speedup for higher values of `X`.
