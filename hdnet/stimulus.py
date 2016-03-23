# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.stimulus
    ~~~~~~~~~~~~~~

    Stimuli time-series.

"""

import os

import numpy as np
from hdnet.util import Restoreable, hdlog
from hdnet.visualization import save_matrix_whole_canvas


class Stimulus(Restoreable, object):
    """ class handling time-series stimuli
    
    Parameters
        stimulus_arr: file_name of data, a numpy M x X array
                    M = number of stimulus (eg. movie) frames
                    X = stimulus array (a movie, etc)

        preprocess: override for other operations on raw data
    """
    _SAVE_ATTRIBUTES_V1 = ['_stimulus_arr', '_M', '_X']
    _SAVE_VERSION = 1
    _SAVE_TYPE = 'Stimulus'

    def __init__(self, stimulus_arr=None, npz_file=None, h5_file=None, preprocess=True):
        """
        Missing documentation
        
        Parameters
        ----------
        stimulus_arr : Type, optional
            Description (default None)
        npz_file : Type, optional
            Description (default None)
        h5_file : Type, optional
            Description (default None)
        preprocess : bool, optional
            Description (default True)
        
        Returns
        -------
        Value : Type
            Description
        """
        object.__init__(self)
        Restoreable.__init__(self)

        # TODO reuse io functionality from data module!

        self.file_name = npz_file or ''
        if npz_file is None and stimulus_arr is None and h5_file is None:
            self._M = 0
            return

        if stimulus_arr is not None:
            self._stimulus_arr = stimulus_arr

        if npz_file is not None:
            if not os.path.isfile(npz_file):
                hdlog.info("File '%s' does not exist!" % npz_file)
                return
            self.file_name = npz_file
            tmp = np.load(npz_file)
            self._stimulus_arr = tmp[tmp.keys()[0]]

        if h5_file is not None:
            import h5py
            f = h5py.File(h5_file)
            self._stimulus_arr = f[f.keys()[0]]

        if preprocess:
            self.preprocess()

        self._M = self._stimulus_arr.shape[0]
        self._X = self._stimulus_arr.shape[1:]

    @property
    def stimulus_arr(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._stimulus_arr

    @property
    def M(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._M

    @property
    def X(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        return self._X

    def preprocess(self):
        """
        Missing documentation
        
        Returns
        -------
        Value : Type
            Description
        """
        pass

    def snapshot(self, start=0, stop=None, save_png_name=None):
        """
        Returns a matrix or saves a PNG of avg of data between start and stop times
        save_png_name: if not None then only saves picture
        
        Parameters
        ----------
        start : int, optional
            Description (default 0)
        stop : Type, optional
            Description (default None)
        save_png_name : Type, optional
            Description (default None)
        
        Returns
        -------
        Value : Type
            Description
        """
        stop = stop or self._M
        sub_stim_arr = self._stimulus_arr[start:stop].mean(axis=0)

        if save_png_name is not None:
            #from PIL import Image
            sub_stim_arr -= 1. * sub_stim_arr.min()
            sub_stim_arr /= sub_stim_arr.max()
            #im_png = Image.fromarray(np.round(255 * sub_stim_arr)).convert('L')
            #im_png.save(save_png_name + '.png')
            save_matrix_whole_canvas(sub_stim_arr, save_png_name + '.png', cmap='gray')
        else:
            return sub_stim_arr

    # i/o

    def save(self, file_name='stimulus', extra=None):
        """
        Saves contents to file.

        Parameters
        ----------
        file_name : str, optional
            File name to save to (default 'stimulus')
        extra : dict, optional
            Extra information to save to file (default None)
        overwrite: bool, optional
            Overwrite flag, whether to overwrite existing files (default False)

        Returns
        -------
        Nothing
        """
        return super(Stimulus, self)._save(file_name=file_name,
                                         attributes=self._SAVE_ATTRIBUTES_V1, version=self._SAVE_VERSION,
                                         extra=extra)

    @classmethod
    def load(cls, file_name='stimulus', load_extra=False):
        """
        Loads contents from file.

        .. note:

            This is a class method, i.e. loading should be done like
            this:

            stimulus = Stimulus.load('file_name')

        Parameters
        ----------
        file_name : str, optional
            File name to load from (default 'stimulus')
        load_extra : bool, optional
            Flag whether to load extra file contents, if any (default False)

        Returns
        -------
        spikes : :class:`.Stimulus`
            Instance of :class:`.Stimulus` if loaded, `None` upon error
        """
        return super(Stimulus, cls)._load(file_name=file_name, load_extra=load_extra)

    def _load_v1(self, contents, load_extra=False):
        # internal function to load v1 file format
        hdlog.debug('Loading Stimulus, format version 1')
        return Restoreable._load_attributes(self, contents, self._SAVE_ATTRIBUTES_V1)

    # representation

    def __repr__(self):
        return '<Stimulus: dimensions {m} x {x}>'.format(m=self.M, x=self.X)

# end of source
