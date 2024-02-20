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

from pathlib import Path

from tests.helpers.utils import create_simulations

from importlib.util import find_spec

import numpy as np
import hist

from boris.io import (
    SimInfo,
    centers_to_edges,
    get_calibration_from_file,
    get_filetype,
    get_keys_in_container,
    obj_resolve_name,
    get_simulation_spectra,
    read_dat_file,
    read_spectrum,
    write_specs,
)


def test_centers_to_edges():
    edges = np.linspace(0, 10, 11)
    centers = 0.5 * (edges[1:] + edges[:-1])
    res = centers_to_edges(centers)
    assert np.all(res == edges)


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_specs_regular_double(tmp_path, filename):
    h = hist.Hist.new.Regular(100, 0, 100).Double(data=np.random.uniform(size=100))
    write_specs(tmp_path / filename, {"testhist": h})
    assert (tmp_path / filename).exists()
    res_hist = read_spectrum(tmp_path / filename, "testhist")
    assert np.isclose(h.values(), res_hist.values()).all()
    assert np.isclose(h.axes[0].edges, res_hist.axes[0].edges).all()


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(
            ("test.root"),
            marks=pytest.mark.skip(
                "Bug in numpy: https://github.com/numpy/numpy/issues/12264"
            ),
        ),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_specs_regular_int(tmp_path, filename):
    h = hist.Hist.new.Regular(100, 0, 100).Int64(data=np.random.uniform(size=100))
    write_specs(tmp_path / filename, {"testhist": h})
    assert (tmp_path / filename).exists()
    res_hist = read_spectrum(tmp_path / filename, "testhist")
    assert np.isclose(h.values(), res_hist.values()).all()
    assert np.isclose(h.axes[0].edges, res_hist.axes[0].edges).all()


def test_get_calibration_from_file(tmp_path):
    with open(tmp_path / "calibration.cal", "w") as f:
        print("abc.txt: 3. 4. -0.5", file=f)
        print("def.txt: 2. 4. -0.5", file=f)
        print("ghi.txt: 1. 4. -0.5", file=f)

    cal = get_calibration_from_file(tmp_path / "calibration.cal", "def.txt")
    assert np.all(cal == np.array([2.0, 4.0, -0.5]))

    with pytest.raises(KeyError):
        cal = get_calibration_from_file(tmp_path / "calibration.cal", "jkl.txt")


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_specs_variable(tmp_path, filename):
    edges = np.linspace(0.0, 100.0, 101)
    h = hist.Hist.new.Variable(edges).Double(data=np.random.uniform(size=100))
    write_specs(tmp_path / filename, {"testhist": h})
    assert (tmp_path / filename).exists()
    res_hist = read_spectrum(tmp_path / filename, "testhist")
    assert np.isclose(h.values(), res_hist.values()).all()
    assert np.isclose(h.axes[0].edges, res_hist.axes[0].edges).all()


def test_write_specs_exists(tmp_path):
    h = hist.Hist.new.Regular(100, 0, 100).Double(data=np.random.uniform(size=100))
    with pytest.raises(Exception):
        write_specs(tmp_path, {"test": h})


def test_write_specs_mkdir(tmp_path):
    h = hist.Hist.new.Regular(100, 0, 100).Double(data=np.random.uniform(size=100))
    write_specs(tmp_path / "dir" / "test.npz", {"test": h})
    assert (tmp_path / "dir" / "test.npz").exists()


def test_write_specs_unknown_format(tmp_path):
    h = hist.Hist.new.Regular(100, 0, 100).Double(data=np.random.uniform(size=100))
    with pytest.raises(Exception):
        write_specs(tmp_path / "unknown.invalid", {"test": h})


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        # ("test.txt"),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_specs_1D(tmp_path, filename):
    hists = {
        f"hist{i}": hist.Hist.new.Regular(100, 0, 100).Double(
            data=np.random.uniform(size=100)
        )
        for i in range(4)
    }
    write_specs(tmp_path / filename, hists)
    assert (tmp_path / filename).exists()


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.npz"),
        pytest.param(
            "test.hdf5",
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_hists_2D(tmp_path, filename):
    hists = {
        f"hist{i}": hist.Hist.new.Regular(100, 0, 100)
        .Regular(100, 0, 100)
        .Double(data=np.random.uniform(size=(100, 100)))
        for i in range(4)
    }
    write_specs(tmp_path / filename, hists)
    assert (tmp_path / filename).exists()


@pytest.mark.parametrize(
    ("filename", "mimetype"),
    [
        ("test.root", "application/root"),
        ("test.npz", "application/zip"),
        # ("test.txt", None),
        pytest.param(
            "test.hdf5",
            "application/x-hdf5",
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_get_filetype(tmp_path, filename, mimetype):
    h = hist.Hist.new.Regular(10, 0, 10).Double(data=np.random.uniform(size=10))
    write_specs(tmp_path / filename, {"test": h})
    assert (tmp_path / filename).exists()
    assert get_filetype(tmp_path / filename) == mimetype


@pytest.mark.parametrize(
    ("delimiter", "mimetype"),
    [
        (" ", "text/space-separated-values"),
        (",", "text/comma-separated-values"),
        ("\t", "text/tab-separated-values"),
    ],
)
def test_get_filetype_xsv(tmp_path, delimiter, mimetype):
    h = np.random.uniform(0, 100, size=(10, 2))
    np.savetxt(tmp_path / "test.xsv", h, delimiter=delimiter)
    assert get_filetype(tmp_path / "test.xsv") == mimetype


def test_get_filetype_plain(tmp_path):
    h = np.random.uniform(0, 100, size=(10,))
    np.savetxt(tmp_path / "spec.txt", h, header="more information")
    assert get_filetype(tmp_path / "spec.txt") == "text/plain"


def test_get_filetype_none(tmp_path):
    with open(tmp_path / "invalid", "w") as f:
        print("abc\ndef", file=f)
    assert get_filetype(tmp_path / "invalid") is None


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_get_keys_in_container(tmp_path, filename):
    hists = {f"hist{i}": np.random.uniform(size=100) for i in range(4)}
    hists = {
        f"hist{i}": hist.Hist.new.Regular(100, 0, 100).Double(
            data=np.random.uniform(size=100)
        )
        for i in range(4)
    }
    write_specs(tmp_path / filename, hists)
    assert (tmp_path / filename).exists()
    keys = get_keys_in_container(tmp_path / filename)
    if filename[-5:] == ".npz":
        for i in range(4):
            assert "hist{i}_nbins" in keys
            assert "hist{i}_start" in keys
            assert "hist{i}_stop" in keys
    for key in hists.keys():
        assert key in keys


def test_get_keys_in_container_xsv(tmp_path):
    h = np.random.uniform(0, 100, size=(10, 2))
    np.savetxt(tmp_path / "test.csv", h, delimiter=",", header="centers,counts")
    keys = get_keys_in_container(tmp_path / "test.csv")
    assert "centers" in keys
    assert "counts" in keys


def test_get_keys_in_container_unsupported(tmp_path):
    np.savetxt(tmp_path / "test.npy", np.array([1, 2, 3]))
    assert (tmp_path / "test.npy").exists()
    res = get_keys_in_container(tmp_path / "test.npy")
    assert len(res) == 0


def test_read_spectrum_txt(tmp_path):
    h = np.random.uniform(0, 100, size=(10,))
    np.savetxt(tmp_path / "test.csv", h)
    read_spectrum(tmp_path / "test.csv")


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
