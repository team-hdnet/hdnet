# -*- coding: utf-8 -*-
"""
    hdnet
    ~~~~~

    Hopfield denoising network setup file

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

from setuptools import setup

setup(
    name='hdnet',
    version=__version__,
    description='Hopfield denoising network',
    url='http://github.com/',
    author='Christopher Hillar, Felix Effenberger',
    author_email='chillar@msri.org, felix.effenberger@mis.mpg.de',
    license='GPLv3',
    packages=['hdnet'],
    install_requires=[
        'numpy',
        'matplotlib',
        'bitstring',
        'networkx',
	'statsmodels',
	'h5py'
    ],
    zip_safe=False)
