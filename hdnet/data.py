# -*- coding: utf-8 -*-
"""
    hdnet.data
    ~~~~~~~~~~

    Data import and transformation functionality.

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

__version__ = "0.1"

__all__ = ()


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
    def read_spikes(path_or_files, rate, discard_first_cluster=True):

        if isinstance(path_or_files, (str, unicode)):
            # glob all res files
            hdlog.info('loading klustakwick data from %s' % os.path.abspath(path_or_files))
            import glob
            res_files = glob.glob(os.path.join(path_or_files, '*.res.*'))
        else:
            res_files = path_or_files
            hdlog.info('loading klustakwick data from files %s' % str(path_or_files))

        hdlog.info('processing %d electrode files' % len(res_files))

        spike_times = []
        all_clusters = set([])
        num_spikes = 0
        t_min = np.inf
        t_max = -np.inf
        electrodes = []

        for fn_res in res_files:
            hdlog.debug('processing electrode file "%s"..' % fn_res)
            electrodes.append(int(fn_res[fn_res.rindex('.') + 1:]))

            fn_clu = fn_res.replace('.res.', '.clu.')
            if not os.path.exists(fn_clu):
                raise Exception('cluster file "%s" not found!' % fn_clu)

            #load time stamps
            times = np.loadtxt(fn_res) * (1. / float(rate))

            #load cluster data
            clusters = np.loadtxt(fn_clu).astype(int)
            n_clusters = clusters[0]
            cluster_seq = clusters[1:]

            if cluster_seq.shape[0] != times.shape[0]:
                raise Exception('data inconsistent for files %s, %s: lengths differ!' % (fn_res, fn_clu))

            hdlog.debug('%d clusters, %d spikes' % (n_clusters, cluster_seq.shape[0]))

            spike_times_electrode = np.array([times[np.where(cluster_seq == i)] for i in xrange(n_clusters) if
                                            not discard_first_cluster or (discard_first_cluster and i > 1)])
            spike_times.append(spike_times_electrode)

            all_clusters.update(cluster_seq)
            num_spikes += sum(map(len, spike_times_electrode))
            t_min = min(t_min, min(times))
            t_max = max(t_max, max(times))

        status = {
            'clusters': list(all_clusters),
            't_min': t_min,
            't_max': t_max,
            'num_spikes': num_spikes,
            'electrodes': electrodes
        }

        hdlog.info('processed %d clusters, %d spikes total, t_min=%f s, t_max=%f s, delta=%f s' %
                   (len(all_clusters) - (2 if discard_first_cluster else 0), num_spikes, t_min, t_max, t_max - t_min))

        return spike_times, status


class Binner(object):
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def bin_spike_times(spike_times, bin_size, t_min=None, t_max=None):
        if t_min is None or t_max is None:
            t_min_dat = np.inf
            t_max_dat = -np.inf
            for cluster_times in spike_times:
                cluster_times_nonempty = [x for x in cluster_times if len(x) > 0]
                t_min_dat = min([t_min_dat] + map(min, cluster_times_nonempty))
                t_max_dat = max([t_max_dat] + map(max, cluster_times_nonempty))

            if t_min is None:
                t_min = t_min_dat

            if t_max is None:
                t_max = t_max_dat

        bins = np.arange(t_min, t_max, bin_size)
        binned = np.zeros((sum(map(len, spike_times)), len(bins)), dtype=int)

        hdlog.info('binning {c} cells between t_min={m} and t_max={M}, {bins} bins'.format(
            c=binned.shape[0], m=t_min, M=t_max, bins=len(bins)
        ))

        pos = 0
        for i in xrange(len(spike_times)):
            for cluster_times in spike_times[i]:
                if len(cluster_times) > 0:
                    indices = np.digitize(cluster_times, bins) - 1
                    binned[pos, indices] = 1
                pos += 1

        return binned


