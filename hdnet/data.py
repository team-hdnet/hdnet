# -*- coding: utf-8 -*-
"""
    hdnet.data
    ~~~~~~~~~~

    Data import and transformation functionality.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

import numpy as np
import os
from util import hdlog


class Reader(object):
    def __init__(self):
        object.__init__(self)

    def read_spikes(self, *args, **kwargs):
        raise NotImplementedError('Reader class is abstract')


class KlustaKwickReader(Reader):
    """ Class for reading KlustaKwick data sets
        https://github.com/klusta-team/klustakwik/

    Parameters
        path_or_files: path of data set or list of *.res.* files to load
        rate: sampling rate [in Hz]
        discard_first_cluster: discard first cluster, usually taked for unclassified spikes (defualt True)
    """
    def __init__(self):
        Reader.__init__(self)

    @staticmethod
    def read_spikes(path_or_files, rate, first_cluster=2, filter_silent=True, return_status=False):

        if isinstance(path_or_files, (str, unicode)):
            # glob all res files
            hdlog.info('Loading KlustaKwick data from %s' % os.path.abspath(path_or_files))
            import glob
            res_files = glob.glob(os.path.join(path_or_files, '*.res.*'))
        else:
            res_files = path_or_files
            hdlog.info('Loading KlustaKwick data from files %s' % str(path_or_files))

        hdlog.info('Processing %d electrode files' % len(res_files))

        spike_times = []
        num_clusters = 0
        num_spikes = 0
        t_min = np.inf
        t_max = -np.inf
        cells_filtered = 0
        electrodes = []

        for fn_res in res_files:
            hdlog.debug('Processing electrode file "%s"..' % fn_res)
            electrodes.append(int(fn_res[fn_res.rindex('.') + 1:]))

            fn_clu = fn_res.replace('.res.', '.clu.')
            if not os.path.exists(fn_clu):
                raise Exception('Cluster file "%s" not found!' % fn_clu)

            #load time stamps
            times = np.loadtxt(fn_res) * (1. / float(rate))

            #load cluster data
            clusters = np.loadtxt(fn_clu).astype(int)
            n_clusters = clusters[0]
            cluster_seq = clusters[1:]

            if cluster_seq.shape[0] != times.shape[0]:
                raise Exception('Data inconsistent for files %s, %s: lengths differ!' % (fn_res, fn_clu))

            hdlog.debug('%d clusters, %d spikes' % (n_clusters, cluster_seq.shape[0]))

            spike_times_electrode = [times[np.where(cluster_seq == c)[0]]
                                     for c in xrange(first_cluster, n_clusters)]

            if filter_silent:
                c_orig = len(spike_times_electrode)
                spike_times_electrode = [x for x in spike_times_electrode if len(x) > 0]
                c_filtered = c_orig - len(spike_times_electrode)
                cells_filtered += c_filtered

            spike_times.extend(spike_times_electrode)

            num_clusters += n_clusters - first_cluster
            num_spikes += sum(map(len, spike_times_electrode))
            t_min = min(t_min, min(times))
            t_max = max(t_max, max(times))

        status = {
            'clusters': num_clusters,
            'discarded_clusters': first_cluster * len(res_files),
            'filtered': cells_filtered,
            't_min': t_min,
            't_max': t_max,
            'num_spikes': num_spikes,
            'electrodes': electrodes
        }

        hdlog.info('Processed %d clusters (%d discarded), %d cells (%d silent discarded), %d spikes total, t_min=%f s, t_max=%f s, delta=%f s' %
                   (num_clusters, first_cluster * len(res_files), num_clusters - cells_filtered, cells_filtered,
                    num_spikes, t_min, t_max, t_max - t_min))

        if return_status:
            return spike_times, status
        else:
            return spike_times


class Binner(object):
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def bin_spike_times(spike_times, bin_size, t_min=None, t_max=None):
        t_min_dat = np.inf
        t_max_dat = -np.inf

        spike_times_nonempty = [x for x in spike_times if len(x) > 0]
        if len(spike_times_nonempty) > 0:
            t_min_dat = min([t_min_dat] + map(min, spike_times_nonempty))
            t_max_dat = max([t_max_dat] + map(max, spike_times_nonempty))

        if t_min is None:
            t_min = t_min_dat

        if t_max is None:
            t_max = t_max_dat

        if t_min == np.inf or t_max == -np.inf:
            hdlog.info('No spikes!')
            return np.zeros((len(spike_times), 1))

        bins = np.arange(t_min, t_max + bin_size, bin_size)
        binned = np.zeros((len(spike_times), len(bins)), dtype=int)

        hdlog.info('Binning {c} cells between t_min={m} and t_max={M}, {bins} bins'.format(
            c=binned.shape[0], m=t_min, M=t_max, bins=len(bins)
        ))

        pos = 0
        for st in spike_times:
            if len(st) > 0:
                indices = np.digitize(st, bins) - 1
                binned[pos, indices] = 1
                pos += 1

        return binned


# end of source
