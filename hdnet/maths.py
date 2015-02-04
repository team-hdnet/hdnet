# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

"""
    hdnet.math
    ~~~~~~~~~~~~~~

    Miscalleaneous mathematical functions for hdnet.

"""

import numpy as np


def heaviside(X, dtype=None):
    r"""
    Heaviside function: given M x N numpy array, return points-wise Heaviside:
    :math:`H(r)=1 \text{ if } r > 0 \text{ else } H(r)=0`

    Parameters:
        X (array_like): input array
    Returns:
        Array with entries of X heavisided
    """
    H = (np.sign(X).astype(int) + 1) // 2
    if dtype is None:
        return H
    else:
        return H.astype(dtype)


# end of source
