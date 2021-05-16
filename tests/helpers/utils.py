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

"""Helpers for boris tests."""

import sys
import contextlib

import numpy as np

from boris.utils import write_hists, SimInfo


def create_simulations(tmp_path, detectors=None, shift_axis=0.0):
    detectors = detectors or ["det1"]
    simulations = []
    for i in np.linspace(0, 600, 9, dtype=int):
        hist = np.zeros(1000)
        hist[:i] = 500 * np.random.uniform(size=i)
        hist[i] = 50000
        bin_edges = np.linspace(0, 1000, 1001) + shift_axis
        path = tmp_path / f"sim_{i}keV.root"
        hists = {det: hist for det in detectors}
        write_hists(hists, bin_edges, path)
        simulations.append(SimInfo(path, i, 1000000))
    return simulations


@contextlib.contextmanager
def hide_module(module):
    try:
        old = sys.modules[module]
    except BaseException:
        sys.modules[module] = None
        yield
        del sys.modules[module]
        return
    sys.modules[module] = None
    yield
    sys.modules[module] = old
