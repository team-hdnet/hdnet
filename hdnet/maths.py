# -*- coding: utf-8 -*-
"""
    hdnet.math
    ~~~~~~~~~~~~~~

    Miscalleaneous mathematical functions for hdnet

    :copyright: Copyright 2014 the authors, see AUTHORS.
    :license: GPLv3, see LICENSE for details.
"""

import numpy as np


def heaviside(X, dtype=None):
    """ given M x N numpy array, return Heaviside points-wise:
        H(r) = 1 if r > 0; else H(r) = 0
    """
    H = (np.sign(X).astype(int) + 1) // 2
    if dtype is None:
        return H
    else:
        return H.astype(dtype)


# end of source
