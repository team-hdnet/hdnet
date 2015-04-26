# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.visualization
    ~~~~~~~~~~~~~~~~~~~

    Visualization functions for hdnet.

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
    mpl.rcParams.update({'font.size': 22})
except ImportError, e:
    pass


def plot_matrix_whole_canvas(matrix, **kwargs):
    """
    Missing documentation
    
    Parameters
    ----------
    matrix : Type
        Description
    kwargs : Type
        Description
    
    Returns
    -------
    Value : Type
        Description
    """
    plt.axis("off")
    ax = plt.axes([0, 0, 1, 1])
    ax.matshow(matrix, **kwargs)
    return ax


def save_matrix_whole_canvas(matrix, fname, **kwargs):
    """
    Missing documentation
    
    Parameters
    ----------
    matrix : Type
        Description
    fname : Type
        Description
    kwargs : Type
        Description
    
    Returns
    -------
    Value : Type
        Description
    """
    plt.figure()
    plot_matrix_whole_canvas(matrix, **kwargs)
    plt.savefig(fname)
    plt.close


def raster_plot_psth(spikes,
                     trial=0,
                     start_idx=0,
                     stop_idx=None,
                     bin_size=0.002,
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
    Missing documentation
    
    Parameters
    ----------
    spikes : Type
        Description
    trial : int, optional
        Description (default 0)
    start_idx : int, optional
        Description (default 0)
    stop_idx : Type, optional
        Description (default None)
    bin_size : int, optional
        Description (default 0.002)
    hist_bin_size : int, optional
        Description (default 0.005)
    label_x : str, optional
        Description (default 'time [s]')
    label_y_hist_x : str, optional
        Description (default '[Hz]')
    label_y_raster : str, optional
        Description (default 'neuron')
    label_x_hist_y : Type, optional
        Description (default None)
    fig_size : Type, optional
        Description (default None)
    hist_x : bool, optional
        Description (default True)
    hist_y : bool, optional
        Description (default False)
    color_raster : str, optional
        Description (default '#070d0d')
    color_hist : str, optional
        Description (default '#070d0d')
    
    Returns
    -------
    Value : Type
        Description
    """
    """
     Displays as raster plot (optionally with PSTH below) of a group of neurons
    """

    # default paramters
    if fig_size is None:
        fig_size = (8, 4)

    if stop_idx is None:
        stop_idx = spikes.shape[2]

    x = []
    y = []
    for cell_idx in xrange(spikes.shape[1]):
        spikes = np.where(spikes[trial][cell_idx][start_idx:stop_idx] == 1)[0]
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
            weights=[hist_bin_size * spikes.shape[1]] * len(x),
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
            weights=[hist_bin_size * spikes.shape[1]] * len(y),
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
        mark_converged=None,
        label_empirical='raw',
        label_patterns='Hopfield',
        plot_mtas=True):
    """
    Missing documentation
    
    Parameters
    ----------
    empirical : Type
        Description
    patterns : Type
        Description
    color_empirical : str, optional
        Description (default 'g')
    color_pattern : str, optional
        Description (default 'r')
    mark_empirical : Type, optional
        Description (default None)
    mark_converged : Type, optional
        Description (default None)
    plot_mtas : bool, optional
        Description (default True)
    
    Returns
    -------
    Value : Type
        Description
    """
    
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

    plot.plot(xrange(1, len(emp_sort) + 1), emp_sort[::-1], color=color_empirical, lw=3, label=label_empirical)
    plot.plot(xrange(1, len(hop_sort) + 1), hop_sort[::-1], color=color_pattern, lw=3, label=label_patterns)

    if mark_empirical:
        mark_empirical = np.array(mark_empirical) + 1
        plot.scatter(mark_empirical, [emp_sort[::-1][idx - 1] for idx in mark_empirical], marker='x', s=100,
                     edgecolor=color_empirical,
                     facecolor=color_empirical, linewidth=3, alpha=1.0)
    if mark_converged:
        mark_converged = np.array(mark_converged) + 1
        plot.scatter(mark_converged, [hop_sort[::-1][idx - 1] for idx in mark_converged], marker='x',
                     s=100, edgecolor=color_pattern, facecolor=color_pattern, linewidth=3, alpha=1.0)

        if label_empirical is not None and label_patterns is not None:
            plot.legend(loc='upper right')

    plt.tight_layout(pad=0.1)

    if mark_converged:
        # examples of patterns
        hop_mtas = patterns.top_mta_matrices(len(patterns))
        hop_mats = patterns.top_binary_matrices(len(patterns))
        emp_mats = empirical.top_binary_matrices(len(empirical))

        fig2, axs2 = plt.subplots(2 + (1 if plot_mtas else 0), len(mark_converged))
        nullfmt = mpl.ticker.NullFormatter()

        for i in xrange(len(mark_converged)):
            # empirical
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

            if plot_mtas:
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

                axs2[2, i].imshow(hop_mtas[-mark_converged[i]], interpolation='nearest', cmap='gray_r')

        fig2.tight_layout(pad=0.1)
        fig2.subplots_adjust(hspace=.25)

        return fig1, ax1, fig2, axs2

    return fig1, ax1


def plot_memories_distribution_matrix(patterns, trials, t_min=None, t_max=None, p_min=None, p_max=None, v_min=None, v_max=None, cmap='Paired'):
    """
    Missing documentation
    
    Parameters
    ----------
    patterns : Type
        Description
    trials : Type
        Description
    t_min : Type, optional
        Description (default None)
    t_max : Type, optional
        Description (default None)
    p_min : Type, optional
        Description (default None)
    p_max : Type, optional
        Description (default None)
    v_min : Type, optional
        Description (default None)
    v_max : Type, optional
        Description (default None)
    cmap : str, optional
        Description (default 'Paired')
    
    Returns
    -------
    Value : Type
        Description
    """
    trial_len = len(patterns.sequence) / trials

    if p_min is None:
        p_min = 0

    if p_max is None:
        p_max = len(patterns.patterns) - 1

    if t_min is None:
        t_min = 0

    if t_max is None:
        t_max = trial_len - 1

    n_fp = p_max - p_min + 1
    sequence = patterns.sequence.reshape((trials, trial_len))[:, t_min:t_max + 1]

    length = sequence.shape[1]
    dists = np.zeros((n_fp, length), dtype=float)

    for i in xrange(length):
        for fp in sequence[:, i]:
            if fp >= p_min and fp <= p_max:
                dists[fp, i] += 1

    dists /= float(trials)

    if not v_min is None:
        dists[dists < v_min] = 0

    if not v_max is None:
        dists[dists > v_max] = 0

    mtas = np.array([patterns.pattern_to_mta_matrix(i) for i in xrange(p_min, p_max + 1)])
    rawpat = np.array([patterns.pattern_to_binary_matrix(i) for i in xrange(p_min, p_max + 1)])

    mls = []
    mlsraw = []
    avgs = []
    rawavgs = []
    for cidx, column in enumerate(dists.T):
        avg = np.dot(column, mtas)
        avgs.append(avg)

        rawavg = np.dot(column, rawpat)
        rawavgs.append(rawavg)

        ml = np.argmax(column)
        mls.append(mtas[ml])
        mlsraw.append(rawpat[ml])

    fig1 = plt.figure()
    plt.xlabel('window position')
    plt.ylabel('memory')
    plt.matshow(np.atleast_2d(dists), cmap=cmap)
    plt.colorbar()

    return fig1, dists, mtas, avgs, rawavgs, mls, mlsraw


def plot_all_matrices(matrices, file_names, cmap='gray', colorbar=True, vmin=None, vmax=None):
    """
    Missing documentation
    
    Parameters
    ----------
    matrices : Type
        Description
    file_names : Type
        Description
    cmap : str, optional
        Description (default 'gray')
    colorbar : bool, optional
        Description (default True)
    vmin : Type, optional
        Description (default None)
    vmax : Type, optional
        Description (default None)
    
    Returns
    -------
    Value : Type
        Description
    """
    # plot all matrices to files specified
    kwargs = {
        'cmap': cmap
    }
    if vmin is not None:
        kwargs['vmin'] = vmin
    if vmax is not None:
        kwargs['vmax'] = vmax
    for m, fn in zip(matrices, file_names):
        plt.figure()
        plt.matshow(m, **kwargs)
        if colorbar:
            plt.colorbar()
        plt.savefig(fn)
        plt.close()


def combine_windows(windows):
    """
    Missing documentation
    
    Returns
    -------
    Value : Type
        Description
    """
    # combine list of windows, averaging overlapping regions
    windows = np.atleast_3d(windows)
    n = windows.shape[1]
    ws = windows.shape[2]
    l = windows.shape[0] + ws - 1
    combined = np.zeros((n, l), dtype=float)
    for i, w in enumerate(windows):
        combined[:, i:i + ws] += w
    # phase in
    avg = min(ws, windows.shape[0])
    for i in xrange(avg):
        combined[:, i] /= float(i + 1)
    # middle
    combined[:, avg:l + 1 - avg] /= float(windows.shape[0])
    # phase out
    for i in xrange(l + 1 - avg, l):
        combined[:, i] /= float(l - i)
    return combined


def plot_graph(g, nodeval=None, cmap_nodes='cool', cmap_edges='autumn',
               node_vmin=None, node_vmax=None, edge_vmin=None, edge_vmax=None,
               draw_edge_weights=True, edge_weight_format='%.3f'):
    """
    Missing documentation
    
    Parameters
    ----------
    g : Type
        Description
    nodeval : Type, optional
        Description (default None)
    cmap1 : str, optional
        Description (default 'Blues_r')
    cmap2 : str, optional
        Description (default 'bone_r')
    node_vmin : Type, optional
        Description (default None)
    node_vmax : Type, optional
        Description (default None)
    edge_vmin : Type, optional
        Description (default None)
    edge_vmax : Type, optional
        Description (default None)
    
    Returns
    -------
    Value : Type
        Description
    """
    import networkx as nx

    fig = plt.figure()
    pos = nx.spring_layout(g)
    kwargs = {}
    if node_vmin is not None:
        kwargs['node_vmin'] = node_vmin
    if node_vmax is not None:
        kwargs['node_vmax'] = node_vmax
    nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(),
                           node_color='0.8' if nodeval is None else nodeval,
                           node_size=500, alpha=1, with_labels=True,
                           cmap=plt.get_cmap(cmap_nodes), **kwargs)

    kwargs = {}
    if edge_vmin is not None:
        kwargs['edge_vmin'] = edge_vmin
    if edge_vmax is not None:
        kwargs['edge_vmax'] = edge_vmax
    labels = {i: str(i) for i in g.nodes()}
    edge_weights = [g.get_edge_data(*e)['weight'] for e in g.edges()]
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), edge_color='0.4', arrows=True)
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(),
                           edge_color=edge_weights,
                           edge_cmap=plt.get_cmap(cmap_edges), arrows=False, **kwargs)
    nx.draw_networkx_labels(g, pos, labels, font_size=10, font_color='k')
    if draw_edge_weights:
        nx.draw_networkx_edge_labels(g, pos,
                                     {e: edge_weight_format % g.get_edge_data(*e)['weight'] for e in g.edges()},
                                     font_size=10, font_color='k')
    plt.axis('off')
    return fig


# end of source
