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
import hist

from boris.io import write_specs, SimInfo


def create_simulations(tmp_path, detectors=None, shift_axis=0.0):
    detectors = detectors or ["det1"]
    simulations = []
    bin_edges = np.linspace(0, 1000, 1001) + shift_axis
    for i in np.linspace(0, 600, 9, dtype=int):
        path = tmp_path / f"sim_{i}keV.root"
        spec = np.zeros(1000)
        spec[:i] = 500 * np.random.uniform(size=i)
        spec[i] = 50000
        spec = hist.Hist.new.Variable(bin_edges).Double(data=spec)
        hists = {det: spec for det in detectors}
        write_specs(path, hists)
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
