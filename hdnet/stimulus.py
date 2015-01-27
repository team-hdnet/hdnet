# -*- coding: utf-8 -*-
"""
    hdnet.spikes_model
    ~~~~~~~~~~~~~~~~~~

    Class handling stimuli time-series.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os

import numpy as np
from hdnet.visualization import save_matrix_whole_canvas


class Stimulus(object):
    """ class handling time-series stimuli
    
    Parameters
        stimulus_arr: filename of data, a numpy M x X array
                    M = number of stimulus (eg. movie) frames
                    X = stimulus array (a movie, etc)

        preprocess: override for other operations on raw data
    """

    def __init__(self, stimulus_arr=None, npz_file=None, h5_file=None, preprocess=True):
        self.file_name = npz_file or ''

        if npz_file is None and stimulus_arr is None and h5_file is None:
            self.M = 0
            return

        if stimulus_arr is not None:
            self.stimulus_arr = stimulus_arr

        if npz_file is not None:
            if os.path.isfile(npz_file):
                print('File found. Loading %s') % npz_file
                self.file_name = npz_file
                tmp = np.load(npz_file)
                self.stimulus_arr = tmp[tmp.keys()[0]]

        if h5_file is not None:
            import h5py

            f = h5py.File(h5_file)
            self.stimulus_arr = f[f.keys()[0]]

        if preprocess:
            self.preprocess()

        self.M = self.stimulus_arr.shape[0]
        self.X = self.stimulus_arr.shape[1:]

    def preprocess(self):
        pass

    def snapshot(self, start=0, stop=None, save_png_name=None):
        """ Returns a matrix or saves a PNG of avg of data between start and stop times
            save_png_name: if not None then only saves (PIL needs to be installed) """
        stop = stop or self.M
        sub_stim_arr = self.stimulus_arr[start:stop].mean(axis=0)

        if save_png_name is not None:
            #from PIL import Image
            sub_stim_arr -= 1. * sub_stim_arr.min()
            sub_stim_arr /= sub_stim_arr.max()
            #im_png = Image.fromarray(np.round(255 * sub_stim_arr)).convert('L')
            #im_png.save(save_png_name + '.png')
            save_matrix_whole_canvas(sub_stim_arr, save_png_name + '.png', cmap='gray')
        else:
            return sub_stim_arr


# end of source
