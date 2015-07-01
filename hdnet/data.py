# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.data
    ~~~~~~~~~~

    Data import and transformation functionality for hdnet.
    Contains import / read functionality for a number of different
    formats such as KlustaKwick and Matlab.
    Regarding Matlab files note the two different (incompatible)
    file formats that can be read with :class:`.MatlabReaderLegacy`
    and :class:`.MatlabReaderHDF5`, depending on the version of Matlab
    that you use.

"""

__version__ = "0.1"

import numpy as np
import os

from util import hdlog
from hdnet.spikes import Spikes


class Reader(object):
    """
    Abstract Reader class, all readers inherit from this class.
    """
    def __init__(self):
        object.__init__(self)


class KlustaKwickReader(Reader):
    """
    Class for reading `KlustaKwick <https://github.com/klusta-team/klustakwik>`_ data sets.
    """
    def __init__(self):
        Reader.__init__(self)

    @staticmethod
    def read_spikes(path_or_files, rate, first_cluster=2, filter_silent=True, return_status=False):
        """
        Reader for `KlustaKwick <https://github.com/klusta-team/klustakwik>`_ files.
        
        Parameters
        ----------
        path_or_files : string
            path of data set or list of \*.res.\* files to load
        rate : float
            sampling rate [in Hz]
        discard_first_cluster : integer, optional
            discard first n clusters, commonly used for unclassified spikes (default 2)
        filter_silent : boolean, optional
            filter out clusters that have no spikes (default True)
        return_status : boolean, optional
            if True returns a status dictionary along with data as second return value (default False)

        Returns
        -------
        spikes_times : numpy array
            returns numpy array of spike times in all clusters. Float values represent spike times
            in seconds (i.e. a value of 1.0 represents a spike at time 1s)
        """

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


class MatlabReaderLegacy(Reader):
    """
    Class for reading legacy Matlab .mat files created by Matlab
    versions prior to 7.3

    Instantiate with ``reader = MatlabReaderLegacy(file_name)``.
    List available Matlab objects with :meth:`get_keys`.
    Access contents with ``reader['NAME_OF_MATLAB_OBJECT']`` or
    ``reader.get_object('NAME_OF_MATLAB_OBJECT')``.

    .. note::

        This class needs the Python module :mod:`scipy`

    Parameters
    ----------
    file_name : str
        File name of Matlab file
    """
    def __init__(self, file_name):
        Reader.__init__(self)
        self.contents = None
        self.read(file_name)

    def __getitem__(self, item):
        return self.get_object(item)

    def read(self, file_name):
        """
        Reads a Matlab file.

        Parameters
        ----------
        file_name : str
            Name of file to read

        Returns
        -------
        contents : dict (key: object)
            contents of file
        """
        if not os.path.exists(file_name):
            hdlog.warn("File '{}' does not exist!".format(file_name))
            return
        import scipy.io
        self.contents = scipy.io.loadmat(file_name, struct_as_record=True)
        return self.contents

    def get_keys(self):
        """
        Returns names of Matlab objects in file.

        Returns
        -------
        keys : list of str
            List of Matlab objects in file
        """
        if not self.contents:
            return None
        return self.contents.keys()

    def get_objects(self):
        """
        Returns dictionary of all Matlab objects in file.

        Returns
        -------
        objects : dictionary (key: object)
            Dictionary of Matlab objects in file
        """
        return self.contents

    def get_object(self, key):
        """
        Returns a Matlab object with given name from the file
        and ``None`` if no such object exists.

        Parameters
        ----------
        key : str
            Name of Matlab object to retrieve

        Returns
        -------
        object : numpy array
            Matlab object,
            ``None`` if no object with name :attr:`key` exists
        """
        if not self.contents or not key in self.contents:
            return None
        else:
            return self.contents[key]


class MatlabReaderHDF5(Reader):
    """
    Class for reading new Matlab .mat files created by Matlab
    versions >= 7.3.

    Instantiate with ``reader = MatlabReaderHDF5(file_name)``.
    List available Matlab objects with :meth:`get_keys`.
    Access contents with ``reader['NAME_OF_MATLAB_OBJECT']``.
    (The object will be converted to a numpy array. To access
    the underlying HD5 object, call :meth:`get_object_raw`)

    .. note::

        This class needs the Python module :mod:`h5py`

    Parameters
    ----------
    file_name : str
        File name of Matlab file
    """
    def __init__(self, file_name):
        Reader.__init__(self)
        self.file = None
        self.open(file_name)

    def __getitem__(self, item):
        return self.get_object_numpy(item)

    def open(self, file_name):
        """
        Opens a Matlab file of HDF format (version >= 7.3).
        Do not forget to close the file with :meth:`close`
        after reading its contents.

        Parameters
        ----------
        file_name : str
            Name of file to read

        Returns
        -------
        file : :class:`h5py.File` object
            Opened Matlab file
        """
        if not os.path.exists(file_name):
            hdlog.warn("File '{}' does not exist!".format(file_name))
            return
        import h5py
        self.file = h5py.File(file_name)

    def close(self):
        """
        Closes the Matlab file if currently open.

        Returns
        -------
        Nothing
        """
        if self.file:
            self.file.close()
            self.file = None

    def get_hdf5(self):
        """
        Returns underlying HDF5 file object belonging to Matlab file.

        Returns
        -------
        file : hdf5 file
            HDF5 file containing Matlab objects
        """
        return self.file

    def keys(self):
        """
        Returns names of Matlab objects in file, ``None``
        if no file open.

        Returns
        -------
        keys : list of str, or ``None``
            List of Matlab objects in file
        """
        if not self.file:
            return None
        return self.file.keys()

    def get_object_raw(self, key):
        """
        Returns object with given name :attr:`key`
        from the Matlab file in raw h5py representation
        and ``None`` if no file currently open or an
        object with given name does not exist.

        Parameters
        ----------
        key : str
            Name of object to be loaded

        Returns
        -------
        object : h5py object
            Matlab object, or ``None``
        """
        if not self.file:
            return None
        return self.file.get(key)

    def get_object_numpy(self, key):
        """
        Returns object with given name :attr:`key`
        from the Matlab file in numpy format
        and ``None`` if no file currently open or
        an object with given name does not exist.

        Parameters
        ----------
        key : str
            Name of object to be loaded

        Returns
        -------
        object : numpy array or ``None``
            Matlab object as numpy array, or ``None``
        """
        raw = self.get_object_raw(key)
        if raw is None:
            return None
        else:
            return np.array(raw)


class SpkReader(Reader):
    """
    Reader for spk file format. See CRCNS Tim Blanche data set.

    .. note::

        This class needs the :mod:`bitstring` module.

    .. warning::

        During testing we encountered errornous data on some
        Linux 64 bit installations. Take care.

    """

    @staticmethod
    def read_spk_files(spk_files, bin_size=1):
        """
        Loads spike times from a list of spk files.
        The j-th item in the list corresponds to the j-th neuron.
        It is the 1d array of spike times (microsec) for that neuron.

        Parameters
        ----------
        spk_files : list of str
            List of strings containing spk file names
        bin_size : int, optional
            Bin size in milliseconds (default 1)

        Returns
        -------
        spikes : numpy array
            numpy array containing binned spike times
        """
        from bitstring import Bits
        neuron_to_file = []
        time_stamps = []
        bin_size = bin_size or 1

        for fn in spk_files:
            neuron_to_file.append(fn)
            f = open(fn, 'rb')
            p = Bits(f)
            fmt = str(p.length / 64) + ' * (intle:64)'
            time_stamps.append(p.unpack(fmt))
        spikes = SpkReader.load_from_spikes_times(time_stamps, bin_size=bin_size)
        return Spikes(spikes)

    @staticmethod
    def read_spk_folder(spk_folder, bin_size=1):
        """
        Loads spike times from all spk files in a given folder.
        The j-th item in the list corresponds to the j-th neuron.
        It is the 1d array of spike times (microsec) for that neuron.

        Parameters
        ----------
        spk_folder : str
            Path containing spk file names
        bin_size : int, optional
            Bin size in milliseconds (default 1)

        Returns
        -------
        spikes : numpy array
            numpy array containing binned spike times
        """
        from bitstring import Bits
        neuron_to_file = []
        time_stamps = []
        bin_size = bin_size or 1
        fns = os.listdir(spk_folder)
        for i, fn in enumerate(fns):
            ext = os.path.splitext(fn)[1]
            if ext in ('.spk', ):  # Blanche spike format
                neuron_to_file.append(fn)
                f = open(os.path.join(spk_folder, fn), 'rb')
                p = Bits(f)
                fmt = str(p.length / 64) + ' * (intle:64)'
                time_stamps.append(p.unpack(fmt))
            spikes = SpkReader.load_from_spikes_times(time_stamps, bin_size=bin_size)
            return Spikes(spikes)


    @staticmethod
    def load_from_spikes_times(spike_times_lists, bin_size=1):
        """
        Loads a spike train from a list of arrays of spike times.
        The j-th item in the list corresponds to the j-th neuron.
        It is the 1d array of spike times (microsec) for that neuron.

        Parameters
        ----------
        spike_times_lists : Type
            Description
        bin_size : int, optional
            Bin size in milliseconds (default 1)

        Returns
        -------
        spikes : numpy array
            numpy array containing binned spike times
        """
        if len(spike_times_lists) == 0:
            return np.array([])
        max_millisec = - np.inf
        for spike_times in spike_times_lists:
            milisec = 1. * (spike_times[-1]) / (10 ** 3)
            max_millisec = max(max_millisec, milisec)
        spikes = np.zeros((len(spike_times_lists), np.int(max_millisec) / bin_size + 1))
        for c, spike_times in enumerate(spike_times_lists):
            for spike_time in spike_times:
                a = int(spike_time / (1000. * bin_size))
                if a < spikes.shape[1]:
                    spikes[c, a] = 1
        return spikes


class Binner(object):
    """
    Spike time binner class. Provides methods that take spike times as list of times
    and bin them.

    **Example**:
    Combined use with KlustaKwick reader::

        spikes_times = KlustaKwickReader.read_spikes(DIRECTORY, SAMPLING_RATE)
        # bin to 1ms bins
        binned_spikes_1ms = Binner.bin_spike_times(spikes_times, 0.001)

    """
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def bin_spike_times(spike_times, bin_size, cells = None, t_min=None, t_max=None):
        """
        Bins given spike_times into bins of size bin_size. Spike times
        expected in seconds (i.e. 1.0 for a spike at second 1, 0.5 for a
        spike happening at 500ms).

        Takes optional arguments cells, t_min and t_max that can be used to restrict
        the cell indices (defaults to all cells) and time range
        (default t_min = minimum of all spike times in spike_times,
        default t_max = maximum of all spike times in spike_times).

        Parameters
        ----------
        spike_times : 2d numpy array
            2d array of spike times of cells, cells as rows
        bin_size : float
            bin size to be used for binning (1ms = 0.001)
        cells : array_like, optional
            indices of cells to process (default None, i.e. all cells)
        t_min : float, optional
            time of leftmost bin (default None)
        t_max : float, optional
            time of rightmost bin (default None)

        Returns
        -------
        spikes : :class:`.Spikes`
            Spikes class containing binned spikes.
        """
        t_min_dat = np.inf
        t_max_dat = -np.inf


        spike_times = np.atleast_1d(spike_times)


        if cells is None:
            cells = np.array(range(len(spike_times)))

        spike_times_nonempty = [x for x in spike_times[cells] if len(x) > 0]
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
        binned = np.zeros((len(spike_times[cells]), len(bins)), dtype=int)

        hdlog.info('Binning {c} cells between t_min={m} and t_max={M}, {bins} bins'.format(
            c=binned.shape[0], m=t_min, M=t_max, bins=len(bins)
        ))

        pos = 0
        for st in spike_times:
            if len(st) > 0:
                indices = np.digitize(st, bins) - 1
                binned[pos, indices] = 1
                pos += 1

        return Spikes(spikes=binned)


class SequenceEncoder(object):
    """
    Sequence encoder class. Provides methods that take spike times as list of times
    and extract firing sequences (just preserving the sequence and discarding other
    timing information).

    **Example**:
    Combined use with KlustaKwick reader::

        spikes_times = KlustaKwickReader.read_spikes(DIRECTORY, SAMPLING_RATE)
        # calculate spikes sequence
        spikes_sequence = Binner.get_spike_sequence(spikes_times)

    """
    def __init__(self):
        object.__init__(self)

    @staticmethod
    def get_spike_sequence(spike_times, cells = None, t_min=None, t_max=None):
        """
        Extracts the firing sequence from the given spike times, i.e. a binary
        matrix S of dimension N x M where N is the number of neurons and M the
        total number of spikes in the data set. Each column of S contains exactly
        one non-zero entry, the index of the cell that spiked. Absolute
        spike timing information is discarded, spike order is preserved.

        Takes optional arguments cells, t_min and t_max that can be used to restrict
        the cell indices (defaults to all cells) and time range
        (default t_min = minimum of all spike times in spike_times,
        default t_max = maximum of all spike times in spike_times).

        Parameters
        ----------
        spike_times : array_like
            2d array of spike times of cells
        cells : array_like, optional
            indices of cells to process (default None, i.e. all cells)
        t_min : float, optional
            time of leftmost bin (default None)
        t_max : float, optional
            time of rightmost bin (default None)

        Returns
        -------
        sequence : 2d numpy array of int
            Spike sequence matrix S
        """
        t_min_dat = np.inf
        t_max_dat = -np.inf

        spike_times = np.atleast_2d(spike_times)

        if cells is None:
            cells = np.array(range(len(spike_times)))

        spike_times_nonempty = [x for x in spike_times[cells] if len(x) > 0]
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

        num_cells = len(cells)
        num_spikes = sum(map(len, spike_times_nonempty))
        sequence = np.zeros((num_cells, num_spikes), dtype=int)

        hdlog.info('Extracting sequences for {c} cells between t_min={m} and t_max={M}, {s} spikes'.format(
            c=num_cells, m=t_min, M=t_max, s=num_spikes))

        times = np.array([s for c in spike_times_nonempty for s in c])
        sort_idx = np.argsort(times)
        idxs = np.array([i for i, c in enumerate(spike_times_nonempty) for _ in c])

        for i, c in enumerate(idxs[sort_idx]):
            sequence[i, c] = 1

        return sequence

# end of source
