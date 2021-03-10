# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.util
    ~~~~~~~~~~

    Utility functions for hdnet

"""

from __future__ import print_function

import os
import logging
import numpy as np

# Logging
HDLOGNAME = 'hdnet'
logging.basicConfig(level=logging.INFO)
hdlog = logging.getLogger(HDLOGNAME)


class Restoreable(object):
    #mixing class for saving
    """
    Mixing class for supporting of saving and loading of contents in compressed
    numpy format (numpy.savez). Supports file versioning and type identification.
    """
    _VERSION_FIELD = 'hdnet_version'
    _TYPE_FIELD = 'hdnet_kind'
    _SAVE_TYPE = ''

    def __init__(self):
        object.__init__(self)

    def _save(self, file_name, attributes, version, has_internal=False, folder_name=None,
              internal_objects=None, extra=None, overwrite=False):
        """
        Missing documentation

        Parameters
        ----------
        file_name : Type
            Description
        attributes : Type
            Description
        version : Type
            Description
        has_internal : bool, optional
            Description (default False)
        folder_name : Type, optional
            Description (default None)
        internal_objects : Type, optional
            Description (default None)
        extra : Type, optional
            Description (default None)

        Returns
        -------
        Value : Type
            Description
        """
        base, ext = os.path.splitext(file_name)
        if not ext:
            ext = ".npz"
        file_name = base + ext

        if has_internal:
            if os.path.exists(folder_name):
                if not overwrite:
                    hdlog.error("Folder '%s' exists, cannot save! Use 'overwrite = True' to overwrite." % folder_name)
                    return
            else:
                os.mkdir(folder_name)
            file_name = os.path.join(folder_name, file_name)

        hdlog.info("Saving to file '%s'" % file_name)

        if extra is None:
            extra = {}

        if file_name is not None:
            for key in attributes:
                if not hasattr(self, key):
                    hdlog.debug("Skipping non-existing attribute '{k}'".format(k=key))
                    value = None
                else:
                    value = getattr(self, key)
                if isinstance(value, dict):
                    extra['DICT_KEYS_' + key] = tuple(value.keys())
                    extra['DICT_VALUES_' + key] = tuple(value.values())
                else:
                    extra[key] = value
            extra[Restoreable._VERSION_FIELD] = version
            extra[Restoreable._TYPE_FIELD] = self._SAVE_TYPE
            hdlog.debug("Kind '%s'" % extra[Restoreable._TYPE_FIELD])
            np.savez(file_name, **extra)

        if has_internal:
            for cls, attr_name, fn in internal_objects:
                attr = getattr(self, attr_name)
                if attr is not None:
                    attr.save(os.path.join(folder_name, fn))

    @classmethod
    def _load(cls, file_name, has_internal=False, folder_name=None, internal_objects=None, load_extra=False):
        """
        Missing documentation

        Parameters
        ----------
        file_name : Type
            Description
        has_internal : bool, optional
            Description (default False)
        folder_name : Type, optional
            Description (default None)
        internal_objects : Type, optional
            Description (default None)
        load_extra : bool, optional
            Description (default False)

        Returns
        -------
        Value : Type
            Description
        """
        base, ext = os.path.splitext(file_name)
        if not ext:
            ext = ".npz"
        file_name = base + ext

        if has_internal:
            file_name = os.path.join(folder_name, file_name)

        if file_name is not None and not os.path.exists(file_name):
            hdlog.info("File '%s' does not exist!" % file_name)
            return None

        instance = cls()

        if file_name is not None:
            hdlog.info("Loading from file '%s'" % file_name)
            contents = Restoreable._load_raw(file_name)

            if not Restoreable._VERSION_FIELD in contents or not Restoreable._TYPE_FIELD in contents:
                hdlog.error("File does not seem to be a valid hdnet data file, missing version / type!")
                return None

            if not contents[Restoreable._TYPE_FIELD] == np.array(instance._SAVE_TYPE):
                hdlog.error("File has wrong type: expected '%s', got '%s'" % (
                    instance._SAVE_TYPE, str(contents[Restoreable._TYPE_FIELD])))
                return None

            loader_name = '_load_v' + str(contents[Restoreable._VERSION_FIELD])
            if not hasattr(instance, loader_name):
                hdlog.error("Class '%s' does not have a loader for file version %d!" % (
                    instance.__class__.__name__, contents[Restoreable._VERSION_FIELD]))
                return None

            loader = getattr(instance, loader_name)
            loader(contents, load_extra=load_extra)

        if has_internal:
            if not os.path.exists(folder_name):
                hdlog.error("Folder '%s' does not exist!" % folder_name)
                return None

            hdlog.info("Loading internal objects")
            for cls, attr_name, file_name in internal_objects:
                full_file_name = os.path.join(folder_name, file_name)
                if os.path.isdir(full_file_name) or os.path.isfile(full_file_name) \
                        or os.path.isfile(full_file_name + '.npz'):
                    setattr(instance, attr_name, cls.load(full_file_name))

        return instance

    @staticmethod
    def _load_attributes(instance, contents, attributes):
        """
        Missing documentation

        Parameters
        ----------
        instance : Type
            Description
        contents : Type
            Description
        attributes : Type
            Description

        Returns
        -------
        Value : Type
            Description
        """
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
    def _load_raw(file_name):
        """
        Missing documentation

        Returns
        -------
        Value : Type
            Description
        """
        base, ext = os.path.splitext(file_name)
        if not ext:
            ext = ".npz"
        file_name = base + ext
        contents = np.load(file_name)
        data = {key: contents[key] for key in contents.keys()}
        contents.close()
        return data


# end of source
