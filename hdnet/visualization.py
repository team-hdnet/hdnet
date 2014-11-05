# -*- coding: utf-8 -*-
"""
    hdnet.visualization
    ~~~~~~~~~~~~~~~~~~~

    Visualization functions for hdnet

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


HAS_PRETTYPLOTLIB = False
try:
    import prettyplotlib as ppl

    HAS_PRETTYPLOTLIB = True
except ImportError, e:
    pass


def raster_plot_psth(spikes_arr,
                     trial=0,
                     start_idx=0,
                     stop_idx=None,
                     bin_size=0.001,
                     hist_bin_size=0.005,
                     label_x='time [s]',
                     label_y_hist_x='[Hz]',
                     label_y_raster='neuron',
                     label_x_hist_y=None,
                     fig_size=None,
                     hist_x=True,
                     hist_y=False,
                     color_raster='#070d0d',
                     color_hist='#070d0d'):
    """
     Displays as raster plot (optionally with PSTH below) of a group of neurons
    """

    # default paramters
    if fig_size is None:
        fig_size = (8, 4)

    if stop_idx is None:
        stop_idx = spikes_arr.shape[2]

    x = []
    y = []
    for cell_idx in xrange(spikes_arr.shape[1]):
        spikes = np.where(spikes_arr[trial][cell_idx][start_idx:stop_idx] == 1)[0]
        x.extend(spikes * bin_size)
        y.extend([cell_idx] * spikes.shape[0])

    x_span = max(x) - min(x)
    hist_bins = int(x_span / hist_bin_size)

    # definitions for the axes
    left = 0.1
    bottom = 0.18
    scatter_width = 0.85
    scatter_height = 0.65
    spacing = 0.05
    bottom_histx = scatter_height + bottom + 0.02
    left_histy = scatter_width + left + 0.02

    height_histx = 0.1
    width_histy = 0.2

    if hist_x:
        # histogram below scatter
        rect_histx = [left, bottom, scatter_width, height_histx]
        rect_scatter = [left, bottom + height_histx + spacing, scatter_width, scatter_height]
        rect_histy = [left_histy, bottom, width_histy, scatter_height]
    else:
        if hist_y:
            # no hist_x, hist_y
            rect_scatter = [left, bottom,
                            scatter_width - spacing - width_histy, scatter_height + height_histx + spacing]
        else:
            # no hist_x, no hist_y
            rect_scatter = [left, bottom, scatter_width, scatter_height + height_histx + spacing]

    # create figure
    fig = plt.figure(1, figsize=fig_size)

    # scatter plot
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.locator_params(nbins=4)
    if HAS_PRETTYPLOTLIB:
        ppl.utils.remove_chartjunk(ax_scatter, ['top', 'right'], show_ticks=False)

    # scatter plot of spikes
    ax_scatter.scatter(x, y, marker='.', color=color_raster, lw=0)

    # set limits
    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)

    ax_scatter.set_xlim((xmin, xmax))
    ax_scatter.set_ylim((ymin, ymax))

    if label_y_raster:
        ax_scatter.set_ylabel(label_y_raster)

    if label_x and not hist_x:
        ax_scatter.set_xlabel(label_x)

    if hist_x:
        ax_scatter.xaxis.set_major_formatter(NullFormatter())
        ax_hist_x = plt.axes(rect_histx)
        ax_hist_x.locator_params(nbins=2)
        if HAS_PRETTYPLOTLIB:
            ppl.utils.remove_chartjunk(ax_hist_x, ['top', 'right'], show_ticks=False)

        ax_hist_x.set_xlim(ax_scatter.get_xlim())
        ax_hist_x.set_xlabel(label_x)
        if label_y_hist_x:
            ax_hist_x.set_ylabel(label_y_hist_x)

        nx, binsx, patchesx = ax_hist_x.hist(
            x, bins=hist_bins,
            weights=[hist_bin_size * spikes_arr.shape[1]] * len(x),
            facecolor=color_hist, ec='none')

    else:
        ax_hist_x = None

    if hist_y:
        ax_hist_y = plt.axes(rect_histy)
        # no labels
        ax_hist_y.yaxis.set_major_formatter(NullFormatter())
        ax_hist_y.locator_params(nbins=2)
        if HAS_PRETTYPLOTLIB:
            ppl.utils.remove_chartjunk(ax_hist_y, ['top', 'right'], show_ticks=False)

        ax_hist_y.set_ylim(ax_scatter.get_ylim())
        ny, binsy, patchesy = ax_hist_y.hist(
            y, bins=hist_bins,
            weights=[hist_bin_size * spikes_arr.shape[1]] * len(y),
            orientation='horizontal', facecolor=color_hist, ec='none')

        if label_x_hist_y:
            ax_hist_y.set_xlabel(label_x_hist_y)
    else:
        ax_hist_y = None

    return fig, ax_scatter, ax_hist_x, ax_hist_x, ax_hist_y


class Visualization(object):
    def __init__(self):
        object.__init__(self)
