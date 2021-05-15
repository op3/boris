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
from importlib.util import find_spec
from unittest import mock

import numpy as np

import boris.makematrix
from boris.utils import write_hist, write_hists, read_spectrum
from boris.makematrix import (
    SimInfo,
    SimSpec,
    read_dat_file,
    interpolate_grid,
    create_matrix,
    make_matrix,
    main,
    init,
)


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


def test_SimSpec(tmp_path):
    hist = np.ones(10)
    bin_edges = np.linspace(0.0, 10.0, 11)
    path = tmp_path / "test0.txt"
    detector = "det1"
    write_hist(path, detector, hist, bin_edges)
    spec = SimSpec(path, detector, 4000, 1000000, 1e3, True)
    assert np.isclose(spec.orig_spec, hist).all()
    assert np.isclose(spec.bin_edges, bin_edges * 1e3).all()
    assert spec.bin_centers.shape[0] + 1 == bin_edges.shape[0]
    assert np.isclose(spec.spec * 1000000, hist).all()

    spec = SimSpec(path, detector, 4000, 1000000, 1e3, False)
    assert np.isclose(spec.spec, hist).all()

    assert spec.binning_convention() == 0.0

    hist = np.ones(10001)
    bin_edges = np.linspace(0, 10.001, 10002) - 0.0005
    path = tmp_path / "test1.txt"
    write_hist(path, detector, hist, bin_edges)
    spec = SimSpec(path, detector, 4000, 1000000, 1e3, False)
    assert np.isclose(spec.binning_convention(), 0.5)


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


def test_create_matrix(tmp_path):
    simulations = create_simulations(tmp_path)

    mat, bin_edges, bin_edges2 = create_matrix(simulations, "det1", scale=1.0)
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 600
    assert np.isclose(bin_edges[0], 0.0)
    assert np.isclose(bin_edges[-1], 600.0)

    mat, bin_edges, bin_edges2 = create_matrix(
        simulations, "det1", scale=1.0, max_energy=1000.0
    )
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 1000


def test_create_matrix_tv(tmp_path):
    simulations = create_simulations(
        tmp_path, detectors=["det2"], shift_axis=-0.5
    )
    mat, bin_edges, bin_edges2 = create_matrix(simulations, "det2", scale=1.0)
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 601
    assert np.isclose(bin_edges[0], -0.5)
    assert np.isclose(bin_edges[-1], 600.5)


@pytest.mark.parametrize(
    "filename",
    [
        ("rema.root"),
        ("rema.npz"),
        pytest.param(
            ("rema.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_make_matrix(tmp_path, filename):
    detectors = ["det1", "det2"]
    simulations = create_simulations(tmp_path, detectors)

    path = tmp_path / "sim.dat"
    with open(path, "w") as f:
        for sim in simulations:
            print(sim.to_dat_file_line(), file=f, end="\n")

    make_matrix(
        path,
        tmp_path / filename,
        detectors,
        sim_dir=Path("/"),
        scale_hist_axis=1.0,
    )
    assert (tmp_path / filename).exists()
    for det in detectors:
        mat, (bin_edges, bin_edges2) = read_spectrum(tmp_path / filename, det)
        assert (bin_edges == bin_edges2).all()
        assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 600
        assert np.isclose(bin_edges[0], 0.0)
        assert np.isclose(bin_edges[-1], 600.0)


def test_makematrix_help():
    sys.argv = ["boris", "--help"]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_makematrix_call(tmp_path):
    simulations = create_simulations(tmp_path)

    path = tmp_path / "sim.dat"
    with open(path, "w") as f:
        for sim in simulations:
            print(sim.to_dat_file_line(), file=f, end="\n")

    sys.argv = [
        "boris",
        "--detector",
        "det1",
        "--scale-hist-axis",
        "1.",
        str(path),
        str(tmp_path / "test.root"),
    ]
    main()


@mock.patch.object(boris.makematrix, "main")
@mock.patch.object(boris.makematrix, "__name__", "__main__")
def test_makematrix_init(main):
    init()
    assert main.called
