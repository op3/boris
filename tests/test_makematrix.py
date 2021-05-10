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

import sys
from pathlib import Path

import numpy as np

from boris.makematrix import SimInfo, read_dat_file, interpolate_grid


def test_SimInfo_from_line():
    res = SimInfo.from_dat_file_line("test.root: 1332.0 1000", Path("/"))

    assert res.path == Path("/test.root")
    assert res.energy == 1332.0
    assert res.nevents == 1000


def test_read_datfile(tmp_path):
    energies = np.linspace(0, 1000, 101)
    with open(tmp_path / "test.dat", "w") as f:
        for i, energy in enumerate(energies):
            nevents = 1000000 + i
            print(f"sim_{i}.root: {energy} {nevents}", file=f)

    res = read_dat_file(tmp_path / "test.dat", Path("/"))

    assert len(res) == 101
    for i, energy in enumerate(energies):
        assert res[i].energy == energy
        assert res[i].nevents == 1000000 + i
        assert res[i].path == Path(f"/sim_{i}.root")


def test_interpolate_grid_midpoint():
    grid = np.linspace(10, 110, 101)
    res = interpolate_grid(grid, 50.3)
    assert len(res) == 2
    assert res[0][:2] == (40, 50)
    assert np.isclose(res[0][2], 0.7)
    assert res[1][:2] == (41, 51)
    assert np.isclose(res[1][2], 0.3)


def test_interpolate_grid_before():
    grid = np.linspace(10, 110, 101)
    res = interpolate_grid(grid, 5)
    assert len(res) == 1
    assert res[0] == (0, 10, 1.0)


def test_interpolate_grid_after():
    grid = np.linspace(10, 110, 101)
    res = interpolate_grid(grid, 120)
    assert len(res) == 1
    assert res[0] == (100, 110, 1.0)
