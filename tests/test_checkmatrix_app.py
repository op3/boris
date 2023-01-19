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

from boris.io import write_specs
from boris.checkmatrix_app import CheckMatrixApp, init


@mock.patch("matplotlib.pyplot")
def test_CheckMatrixApp_plot(mock_plt, tmp_path):
    rema = np.diag(np.ones(10))
    bin_edges = np.linspace(2000, 2200, 11)
    write_hist(tmp_path / "rema.npz", "rema", rema, bin_edges)

    sys.argv = [
        "boris2spec",
        "--left=0",
        "--right=5",
        "--binning-factor=1",
        str(tmp_path / "rema.npz"),
    ]
    CheckMatrixApp()
    assert mock_plt.pcolormesh.called
    assert mock_plt.show.called


@mock.patch("boris.checkmatrix_app.CheckMatrixApp")
@mock.patch("boris.checkmatrix_app.__name__", "__main__")
def test_app_init_CheckMatrixApp(app):
    init()
    assert app.called
