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

from boris.io import read_spectrum, write_specs
from boris.boris2spec_app import boris2spec_app

pytest.skip(allow_module_level=True)


def test_boris2spec_app(tmp_path):
    incident = np.ones((100, 10))
    other = np.ones(100)
    bin_edges = np.linspace(2000, 2200, 11)
    write_hists(
        {"incident": incident, "spectrum": other},
        bin_edges,
        tmp_path / "incident.npz",
    )

    sys.argv = [
        "boris2spec",
        "--var-names",
        "incident",
        "spectrum",
        "--get-mean",
        "--get-median",
        "--get-variance",
        "--get-std-dev",
        "--get-min",
        "--get-max",
        "--get-hdi",
        str(tmp_path / "incident.npz"),
        str(tmp_path / "output.npz"),
    ]
    boris2spec_app()
    assert (tmp_path / "output.npz").exists()
    mean, (bin_edges,) = read_spectrum(tmp_path / "output.npz", "incident_mean")
    assert np.isclose(mean, np.ones(10)).all()
    assert bin_edges.shape[0] == 11

    for key in ["mean", "median", "var", "std", "hdi_lo", "hdi_hi"]:
        res, (bin_edges,) = read_spectrum(tmp_path / "output.npz", f"incident_{key}")
        assert res.ndim == 1
        assert res.shape[0] == 10


@mock.patch("matplotlib.pyplot.show")
def test_boris2spec_app_plot(mock_plt, tmp_path):
    incident = np.ones((100, 10))
    other = np.ones(10)
    bin_edges = np.linspace(2000, 2200, 11)
    write_hists(
        {"incident": incident, "spectrum": other},
        bin_edges,
        tmp_path / "incident.npz",
    )

    sys.argv = [
        "boris2spec",
        "--var-names",
        "incident",
        "spectrum",
        "--get-mean",
        "--get-hdi",
        "--plot",
        "--",
        str(tmp_path / "incident.npz"),
    ]
    boris2spec_app()
    assert mock_plt.called


@mock.patch("matplotlib.pyplot.savefig")
def test_boris2spec_app_plot_export(mock_plt, tmp_path):
    incident = np.ones((100, 10))
    other = np.ones(10)
    bin_edges = np.linspace(2000, 2200, 11)
    write_hists(
        {"incident": incident, "spectrum": other},
        bin_edges,
        tmp_path / "incident.npz",
    )

    sys.argv = [
        "boris2spec",
        "--var-names",
        "incident",
        "spectrum",
        "--get-mean",
        "--get-hdi",
        "--plot",
        str(tmp_path / "output.png"),
        "--plot-title",
        "Plot Title",
        "--plot-xlabel",
        "Energy (keV)",
        "--plot-ylabel",
        "Counts",
        str(tmp_path / "incident.npz"),
    ]
    boris2spec_app()
    assert mock_plt.called


def test_boris2spec_app_wrong_args(tmp_path):
    sys.argv = ["boris2spec", str(tmp_path / "incident.npz")]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        boris2spec_app()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2

    sys.argv = [
        "boris2spec",
        str(tmp_path / "incident.npz"),
        str(tmp_path / "output.npz"),
    ]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        boris2spec_app()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2
