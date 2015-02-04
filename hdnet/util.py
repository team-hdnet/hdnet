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
    numpy format (savez). Supports file versioning and type identification.
    """
    _VERSION_FIELD = 'hdnet_version'
    _TYPE_FIELD = 'hdnet_kind'

    def __init__(self):
        object.__init__(self)

    def save(self, filename, attributes, version, extra=None):
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext

        hdlog.info("saving to file '%s'" % filename)

        if extra is None:
            extra = {}

        for key in attributes:
            value = getattr(self, key)
            if isinstance(value, dict):
                extra['DICT_KEYS_' + key] = value.keys()
                extra['DICT_VALUES_' + key] = value.values()
            else:
                extra[key] = value
        extra[Restoreable._VERSION_FIELD] = version
        extra[Restoreable._TYPE_FIELD] = self.__class__.__name__
        hdlog.info("kind '%s'" % extra[Restoreable._TYPE_FIELD])
        np.savez(filename, **extra)

    @classmethod
    def load(cls, filename, load_extra=False):
        instance = cls()
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext

        hdlog.info("loading from file '%s'" % filename)
        contents = Restoreable.load_raw(filename)

        if not Restoreable._VERSION_FIELD in contents or not Restoreable._TYPE_FIELD in contents:
            hdlog.error("file does not seem to be a valid hdnet data file, missing version / type!")
            return instance

        if not contents[Restoreable._TYPE_FIELD] == np.array(instance.__class__.__name__):
            hdlog.error("file has wrong type: expected '%s', got '%s'" % (
                instance.__class__.__name__, str(contents[Restoreable._TYPE_FIELD])))
            return instance

        loader_name = 'load_v' + str(contents[Restoreable._VERSION_FIELD])
        if not hasattr(instance, loader_name):
            hdlog.error("class '%s' does not have a loader for file version %d!" % (
                instance.__class__.__name__, contents[Restoreable._VERSION_FIELD]))
            return instance

        loader = getattr(instance, loader_name)
        return loader(contents, load_extra=load_extra)

    @staticmethod
    def load_attributes(instance, contents, attributes):
        for attr in attributes:
            if not attr in contents.keys() and not 'DICT_KEYS_' + attr in contents.keys():
                hdlog.info("attribute '%s' not in file" % attr)
                continue
            if 'DICT_KEYS_' + attr in contents.keys():
                value = dict(zip(contents['DICT_KEYS_' + attr],
                                 contents['DICT_VALUES_' + attr]))
            else:
                value = contents[attr]
            setattr(instance, attr, value)
        return instance

    @staticmethod
    def load_raw(filename):
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = ".npz"
        filename = base + ext
        contents = np.load(filename)
        data = {key: contents[key] for key in contents.keys()}
        contents.close()
        return data

# end of source
