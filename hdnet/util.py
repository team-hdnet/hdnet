# -*- coding: utf-8 -*-
"""
    hdnet.util
    ~~~~~~~~~~

    Utility functions

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import os
import logging
import numpy as np

# Logging
HDLOGNAME = 'hdnet'
logging.basicConfig(level=logging.INFO)
hdlog = logging.getLogger(HDLOGNAME)


class Restoreable(object):
    #mixin class for saving
    """
    Mixin class for supporting of saving and loading of contents in compressed
    numpy format (numpy.savez). Supports file versioning and type identification.
    """
    _VERSION_FIELD = 'hdnet_version'
    _TYPE_FIELD = 'hdnet_kind'
    _SAVE_TYPE = ''

    def __init__(self):
        object.__init__(self)

    def _save(self, filename, attributes, version, extra=None):
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext

        hdlog.info("Saving to file '%s'" % filename)

        if extra is None:
            extra = {}

        for key in attributes:
            if not hasattr(self, key):
                hdlog.debug("Skipping attribute '{k}'".format(k=key))
                value = None
            else:
                value = getattr(self, key)
            if isinstance(value, dict):
                extra['DICT_KEYS_' + key] = value.keys()
                extra['DICT_VALUES_' + key] = value.values()
            else:
                extra[key] = value
        extra[Restoreable._VERSION_FIELD] = version
        extra[Restoreable._TYPE_FIELD] = self._SAVE_TYPE
        hdlog.debug("Kind '%s'" % extra[Restoreable._TYPE_FIELD])
        np.savez(filename, **extra)

    @classmethod
    def _load(cls, filename, load_extra=False):
        instance = cls()
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext

        hdlog.info("Loading from file '%s'" % filename)
        contents = Restoreable._load_raw(filename)

        if not Restoreable._VERSION_FIELD in contents or not Restoreable._TYPE_FIELD in contents:
            hdlog.error("File does not seem to be a valid hdnet data file, missing version / type!")
            return instance

        if not contents[Restoreable._TYPE_FIELD] == np.array(instance._SAVE_TYPE):
            hdlog.error("File has wrong type: expected '%s', got '%s'" % (
                instance._SAVE_TYPE, str(contents[Restoreable._TYPE_FIELD])))
            return instance

        loader_name = '_load_v' + str(contents[Restoreable._VERSION_FIELD])
        if not hasattr(instance, loader_name):
            hdlog.error("Class '%s' does not have a loader for file version %d!" % (
                instance.__class__.__name__, contents[Restoreable._VERSION_FIELD]))
            return instance

        loader = getattr(instance, loader_name)
        return loader(contents, load_extra=load_extra)

    @staticmethod
    def _load_attributes(instance, contents, attributes):
        for attr in attributes:
            if not attr in contents.keys() and not 'DICT_KEYS_' + attr in contents.keys():
                hdlog.info("Attribute '%s' not in file" % attr)
                continue
            if 'DICT_KEYS_' + attr in contents.keys():
                value = dict(zip(contents['DICT_KEYS_' + attr],
                                 contents['DICT_VALUES_' + attr]))
            else:
                value = contents[attr]
            setattr(instance, attr, value)
        return instance

    @staticmethod
    def _load_raw(filename):
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext
        contents = np.load(filename)
        data = {key: contents[key] for key in contents.keys()}
        contents.close()
        return data


# end of source
