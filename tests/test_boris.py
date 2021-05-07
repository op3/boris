#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2020–2021 O. Papst.
#
# This file is part of boris.
#
# boris is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# boris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with boris.  If not, see <http://www.gnu.org/licenses/>.

import pytest

import numpy as np

from boris import rebin_uniform, get_rema, deconvolute


def test_rebin_uniform():
    bin_edges_old = np.linspace(0.0, 50.0, 51)
    bin_edges_new = np.linspace(0.0, 60.0, 21)
    data = np.random.normal(25.0, 7.0, size=1000)
    hist = np.histogram(data, bin_edges_old)[0]
    hist_new = rebin_uniform(hist, bin_edges_old, bin_edges_new)
    assert np.sum(hist) == np.sum(hist_new)


def test_rebin_uniform_negative():
    bin_edges_old = np.linspace(0.0, 50.0, 51)
    bin_edges_new = np.linspace(0.1, 50.1, 51)
    data = np.random.normal(25.0, 7.0, size=1000)
    hist = np.histogram(data, bin_edges_old)[0]
    hist[0] = -1
    with pytest.raises(ValueError):
        rebin_uniform(hist, bin_edges_old, bin_edges_new)


def test_rebin_uniform_non_integer():
    bin_edges_old = np.linspace(0.0, 50.0, 51)
    bin_edges_new = np.linspace(0.1, 50.1, 51)
    data = np.random.normal(25.0, 7.0, size=1000)
    hist = np.histogram(data, bin_edges_old)[0]
    hist = 2.2 * hist.astype("float")
    with pytest.raises(ValueError):
        rebin_uniform(hist, bin_edges_old, bin_edges_new)


def test_rebin_uniform_same():
    bin_edges_old = np.linspace(0.0, 50.0, 51)
    bin_edges_new = np.linspace(-1.0, 52.0, 54)
    data = np.random.normal(25.0, 7.0, size=1000)
    hist = np.histogram(data, bin_edges_old)[0]
    hist_new = rebin_uniform(hist, bin_edges_old, bin_edges_new)
    assert (hist == hist_new[1:-2]).all()
