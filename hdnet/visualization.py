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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HAS_PRETTYPLOTLIB = False
try:
    import prettyplotlib as ppl

    HAS_PRETTYPLOTLIB = True
except ImportError, e:
    pass

HAS_BREWER2MPL = False
try:
    import brewer2mpl

    set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
    mpl.rc({'axes.color_cycle': set2, 'lines.linewidth': .75})
    matplotlib.rcParams.update({'font.size': 22})
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


def pattern_rank_plot(
        empirical,
        patterns,
        color_empirical='g',
        color_pattern='r',
        mark_empirical=None,
        mark_converged=None):
    
    hop_vals = np.array(patterns.counts.values())
    hop_idx = hop_vals.argsort()
    hop_sort = hop_vals[hop_idx]

    emp_vals = np.array(empirical.counts.values())
    emp_idx = emp_vals.argsort()
    emp_sort = emp_vals[emp_idx]

    # converged patterns vs empirical patterns
    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Pattern rank')
    ax1.set_ylabel('# of occurrences')

    if HAS_PRETTYPLOTLIB:
        plot = ppl
    else:
        plot = plt

    plot.plot(xrange(1, len(emp_sort) + 1), emp_sort[::-1], color=color_empirical, lw=3)
    plot.plot(xrange(1, len(hop_sort) + 1), hop_sort[::-1], color=color_pattern, lw=3)

    if mark_empirical:
        mark_empirical = np.array(mark_empirical)+1
        plot.scatter(mark_empirical, [emp_sort[::-1][idx - 1] for idx in mark_empirical], marker='x', s=100, edgecolor=color_empirical,
                     facecolor=color_empirical, linewidth=3, alpha=1.0)
    if mark_converged:
        mark_converged = np.array(mark_converged)+1
        plot.scatter(mark_converged, [hop_sort[::-1][idx - 1] for idx in mark_converged], marker='x',
                     s=100, edgecolor=color_pattern, facecolor=color_pattern, linewidth=3, alpha=1.0)

    plt.tight_layout(pad=0.1)

    if mark_converged:
        # examples of patterns
        hop_stas = patterns.top_sta_matrices(len(patterns))
        hop_mats = patterns.top_binary_matrices(len(patterns))
        emp_mats = empirical.top_binary_matrices(len(empirical))

        fig2, axs2 = plt.subplots(3, len(mark_converged))
        nullfmt = mpl.ticker.NullFormatter()

        for i in xrange(len(mark_converged)):
            #empirical
            axs2[0, i].xaxis.set_major_formatter(nullfmt)
            axs2[0, i].yaxis.set_major_formatter(nullfmt)
            if HAS_PRETTYPLOTLIB:
                ppl.utils.remove_chartjunk(axs2[0, i], [], show_ticks=False)
            if i == 0:
                axs2[0, i].set_ylabel('Original pattern')
                axs2[0, i].yaxis.label.set_color(set2[0])
                axs2[0, i].set_xlabel('Rank %d' % mark_empirical[i])
            else:
                axs2[0, i].set_xlabel('%d' % mark_empirical[i])

            axs2[0, i].imshow(emp_mats[-mark_empirical[i]], interpolation='nearest', cmap='gray_r')

            #converged
            if HAS_PRETTYPLOTLIB:
                ppl.utils.remove_chartjunk(axs2[1, i], [], show_ticks=False)
            axs2[1, i].xaxis.set_major_formatter(nullfmt)
            axs2[1, i].yaxis.set_major_formatter(nullfmt)
            if i == 0:
                axs2[1, i].set_ylabel('Memory pattern')
                axs2[1, i].yaxis.label.set_color(set2[1])
                axs2[1, i].set_xlabel('Rank %d' % mark_converged[i])
            else:
                axs2[1, i].set_xlabel('%d' % mark_converged[i])

            axs2[1, i].imshow(hop_mats[-mark_converged[i]], interpolation='nearest', cmap='gray_r')

            #sta
            if HAS_PRETTYPLOTLIB:
                ppl.utils.remove_chartjunk(axs2[2, i], [], show_ticks=False)
            axs2[2, i].xaxis.set_major_formatter(nullfmt)
            axs2[2, i].yaxis.set_major_formatter(nullfmt)
            if i == 0:
                axs2[2, i].set_ylabel('MTA')
                axs2[2, i].set_xlabel('Rank %d' % mark_converged[i])
            else:
                axs2[2, i].set_xlabel('%d' % mark_converged[i])

            axs2[2, i].imshow(hop_stas[-mark_converged[i]], interpolation='nearest', cmap='gray_r')

        fig2.tight_layout(pad=0.1)
        fig2.subplots_adjust(hspace=.25)

        return fig1, ax1, fig2, axs2

    return fig1, ax1


class Visualization(object):
    def __init__(self):
        object.__init__(self)
