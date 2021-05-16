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
from unittest import mock

import numpy as np

from boris.utils import write_hist
from boris.boris_app import BorisApp, init


def test_BorisApp(tmp_path):
    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        "--burn=50",
        "--bg-spectrum",
        str(tmp_path / "background.npz"),
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed.npz"),
        str(tmp_path / "incident.npz"),
    ]
    rema = 0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2)
    bin_edges = np.linspace(2000, 2200, 11)
    incident = np.random.uniform(10, 1000, size=10).astype(np.int64)
    background = np.random.uniform(10, 100, size=10).astype(np.int64)
    observed = (incident @ rema).astype(np.int64)
    observed = (incident @ rema + background).astype(np.int64)
    write_hist(tmp_path / "rema.npz", "rema", rema, bin_edges)
    write_hist(tmp_path / "observed.npz", "observed", observed, bin_edges)
    write_hist(tmp_path / "background.npz", "background", background, bin_edges)
    BorisApp()
    (tmp_path / "incident.npz").exists()

    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        "--burn=50",
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed.npz"),
        str(tmp_path / "incident_wobg.npz"),
    ]
    BorisApp()
    assert (tmp_path / "incident_wobg.npz").exists()


@mock.patch("boris.boris_app.BorisApp")
@mock.patch("boris.boris_app.__name__", "__main__")
def test_app_init_BorisApp(app):
    init()
    assert app.called
