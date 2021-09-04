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

from boris.utils import (
    SimInfo,
    SimSpec,
    _bin_edges_dict,
    create_matrix,
    get_bin_edges,
    get_filetype,
    get_keys_in_container,
    get_obj_bin_edges,
    get_obj_by_name,
    get_rema,
    hdi,
    interpolate_grid,
    read_dat_file,
    read_pos_int_spectrum,
    read_rebin_spectrum,
    read_spectrum,
    rebin_hist,
    rebin_uniform,
    reduce_binning,
    write_hist,
    write_hists,
)


def test__bin_edges_dict_1D():
    bin_edges = np.linspace(0, 10, 11)
    d = _bin_edges_dict(bin_edges)
    assert "bin_edges" in d
    assert (d["bin_edges"] == bin_edges).all()
    assert len(d) == 1


def test__bin_edges_dict_2D():
    bin_edges = np.linspace(0, 10, 11)
    d = _bin_edges_dict([bin_edges, bin_edges])
    assert "bin_edges_0" in d
    assert "bin_edges_1" in d
    assert (d["bin_edges_0"] == bin_edges).all()
    assert (d["bin_edges_1"] == bin_edges).all()
    assert len(d) == 2


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.txt"),
        ("test.npz"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_hist(tmp_path, filename):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    write_hist(tmp_path / filename, "testhist", hist, bin_edges)
    assert (tmp_path / filename).exists()
    res_hist, (res_bin_edges,) = read_spectrum(tmp_path / filename, "testhist")
    assert np.isclose(hist, res_hist).all()
    assert np.isclose(bin_edges, res_bin_edges).all()


def test_write_hist_exists(tmp_path):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    with pytest.raises(Exception):
        write_hist(tmp_path, "test", hist, bin_edges)


def test_write_hists_exists(tmp_path):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    with pytest.raises(Exception):
        write_hists({"test": hist}, bin_edges, tmp_path)


def test_write_hist_mkdir(tmp_path):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    write_hist(tmp_path / "dir" / "test.npz", "test", hist, bin_edges)
    assert (tmp_path / "dir" / "test.npz").exists()


def test_write_hists_mkdir(tmp_path):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    write_hists({"test": hist}, bin_edges, tmp_path / "dir" / "test.npz")
    assert (tmp_path / "dir" / "test.npz").exists()


def test_write_hist_txt_1D(tmp_path):
    hist = np.random.uniform(size=(10, 10))
    bin_edges = np.linspace(0, 10, 11)
    with pytest.raises(Exception):
        write_hist(tmp_path / "test.txt", "test", hist, bin_edges)


def test_write_hist_unknown_format(tmp_path):
    hist = np.random.uniform(size=(10, 10))
    bin_edges = np.linspace(0, 10, 11)
    with pytest.raises(Exception):
        write_hist(tmp_path / "unknown.invalid", "test", hist, bin_edges)


@pytest.mark.parametrize(
    "filename",
    [
        ("test.root"),
        ("test.npz"),
        ("test.txt"),
        pytest.param(
            ("test.hdf5"),
            marks=pytest.mark.skipif(
                find_spec("h5py") is None, reason="Module h5py not installed"
            ),
        ),
    ],
)
def test_write_hists_1D(tmp_path, filename):
    hists = {f"hist{i}": np.random.uniform(size=100) for i in range(4)}
    bin_edges = np.linspace(0, 100, 101)
    write_hists(hists, bin_edges, tmp_path / filename)
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
    hists = {f"hist{i}": np.random.uniform(size=(100, 200)) for i in range(4)}
    bin_edges = [np.linspace(0, 100, 101), np.linspace(0, 200, 201)]
    write_hists(hists, bin_edges, tmp_path / filename)
    assert (tmp_path / filename).exists()


def test_write_hists_unknown_format(tmp_path):
    hists = {f"hist{i}": np.random.uniform(size=(100, 200)) for i in range(4)}
    bin_edges = [np.linspace(0, 100, 101), np.linspace(0, 200, 201)]
    with pytest.raises(Exception):
        write_hists(hists, bin_edges, tmp_path / "unknown.invalid")


def test_write_hists_txt_2D(tmp_path):
    hists = {f"hist{i}": np.random.uniform(size=(100, 200)) for i in range(4)}
    bin_edges = [np.linspace(0, 100, 101), np.linspace(0, 200, 201)]
    with pytest.raises(Exception):
        write_hists(hists, bin_edges, tmp_path / "test.txt")


@pytest.mark.parametrize(
    ("filename", "mimetype"),
    [
        ("test.root", "application/root"),
        ("test.npz", "application/zip"),
        ("test.txt", None),
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
    hist = np.random.uniform(size=10)
    bin_edges = np.linspace(0, 10, 11)
    write_hist(tmp_path / filename, "test", hist, bin_edges)
    assert (tmp_path / filename).exists()
    assert get_filetype(tmp_path / filename) == mimetype


def test_get_bin_edges_centers():
    hist = np.random.uniform(size=10)
    bin_edges = np.linspace(0, 10, 11)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    res_hist, [res_bin_edges] = get_bin_edges(np.array([bin_centers, hist]))
    assert np.isclose(res_hist, hist).all()
    assert np.isclose(res_bin_edges, bin_edges).all()


def test_get_bin_edges_lo_hi():
    hist = np.random.uniform(size=10)
    bin_edges = np.linspace(0, 10, 11)
    res_hist, [res_bin_edges] = get_bin_edges(
        np.array([bin_edges[:-1], bin_edges[1:], hist])
    )
    assert np.isclose(res_hist, hist).all()
    assert np.isclose(res_bin_edges, bin_edges).all()


def test_get_bin_edges_T():
    hist = np.random.uniform(size=10)
    bin_edges = np.linspace(0, 10, 11)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    res_hist, [res_bin_edges] = get_bin_edges(np.array([bin_centers, hist]).T)
    assert np.isclose(res_hist, hist).all()
    assert np.isclose(res_bin_edges, bin_edges).all()


def test_get_bin_edges_ndim():
    hist = np.random.uniform(size=(10, 3, 3))
    with pytest.raises(ValueError):
        get_bin_edges(hist)

    hist = np.random.uniform(size=(10,))
    with pytest.raises(ValueError):
        get_bin_edges(hist)

    with pytest.raises(ValueError):
        get_bin_edges(np.array([2.0])[0])


def test_get_bin_edges_non_continuous():
    hist = np.random.uniform(size=(10, 3))
    with pytest.raises(ValueError):
        get_bin_edges(hist)


def test_get_obj_by_name_single():
    res = get_obj_by_name({"a": 1})
    assert res == 1


def test_get_obj_by_name_single_ignore_bin_edges():
    res = get_obj_by_name({"a": 1, "bin_edges": 2})
    assert res == 1
    res = get_obj_by_name({"a": 1, "bin_edges_0": 2, "bin_edges_1": 3})
    assert res == 1


def test_get_obj_by_name_name():
    res = get_obj_by_name({"a": 1}, "a")
    assert res == 1


def test_get_obj_by_name_not_found():
    with pytest.raises(KeyError):
        get_obj_by_name({})

    with pytest.raises(KeyError):
        get_obj_by_name({"a": 1, "b": 2})

    with pytest.raises(KeyError):
        get_obj_by_name({"a": 1, "b": 2, "bin_edges": 2})


def test_get_obj_bin_edges_1D():
    res = get_obj_bin_edges({"bin_edges": 1, "a": 3})
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0] == 1


def test_get_obj_bin_edges_2D():
    res = get_obj_bin_edges({"bin_edges_1": 1, "a": 3, "bin_edges_0": 2})
    assert type(res) == list
    assert len(res) == 2
    assert res[0] == 2
    assert res[1] == 1


def test_get_obj_bin_edges_not_found():
    with pytest.raises(KeyError):
        get_obj_bin_edges({"a": 3})


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
    bin_edges = np.linspace(0, 100, 101)
    write_hists(hists, bin_edges, tmp_path / filename)
    assert (tmp_path / filename).exists()
    keys = get_keys_in_container(tmp_path / filename)
    if filename[-5:] != ".root":
        assert "bin_edges" in keys
    for key in hists.keys():
        assert key in keys


def test_get_keys_in_container_unsupported(tmp_path):
    np.savetxt(tmp_path / "test.npy", np.array([1, 2, 3]))
    assert (tmp_path / "test.npy").exists()
    res = get_keys_in_container(tmp_path / "test.npy")
    assert len(res) == 0


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


def test_reduce_binning():
    bin_edges = np.linspace(10.0, 110.0, 11)
    left_bin, right_bin = reduce_binning(bin_edges, 2, 20.0, 100.0)
    assert left_bin == 1
    assert right_bin == 9

    left_bin, right_bin = reduce_binning(bin_edges, 2, 25.0, 95.0)
    assert left_bin == 1
    assert right_bin == 9

    left_bin, right_bin = reduce_binning(bin_edges, 2, 25.0, 85.0)
    assert left_bin == 1
    assert right_bin == 9

    left_bin, right_bin = reduce_binning(bin_edges, 2)
    assert left_bin == 0
    assert right_bin == bin_edges.shape[0] - 1

    bin_edges = np.linspace(0, 32, 33)
    left_bin, right_bin = reduce_binning(bin_edges, 3, 6, 27)
    assert left_bin == 6
    assert right_bin == 27


@pytest.mark.parametrize(
    "size",
    [
        ((32,)),
        ((32, 32)),
        ((32, 32, 32)),
    ],
)
def test_rebin_hist(size):
    hist = np.random.uniform(size=size)
    rebin, bin_edges = rebin_hist(hist, 4)
    assert rebin.shape[0] + 1 == bin_edges.shape[0]
    assert np.isclose(hist.sum(), rebin.sum() * 4)
    for i, j in zip(hist.shape, rebin.shape):
        assert i == 4 * j


@pytest.mark.parametrize(
    "binning_factor,res_bin_edges",
    [
        (2, np.linspace(6.0, 26.0, 11)),
        (3, np.linspace(6.0, 27.0, 8)),
    ],
)
def test_rebin_hist_lr(binning_factor, res_bin_edges):
    hist = np.random.uniform(size=(32, 32))
    rebin, bin_edges = rebin_hist(hist, binning_factor, left=6, right=26)
    assert (np.isclose(bin_edges, res_bin_edges)).all()
    assert rebin.shape[0] + 1 == bin_edges.shape[0]
    assert rebin.shape[1] + 1 == bin_edges.shape[0]


def test_rebin_hist_nonsquare():
    hist = np.random.uniform(size=(32, 16))
    with pytest.raises(ValueError):
        rebin_hist(hist, 2)


@pytest.mark.parametrize(
    "ndim",
    [
        (1),
        (2),
        (3),
    ],
)
def test_rebin_hist_bin_edges(ndim):
    hist = np.random.uniform(size=(32,) * ndim)
    bin_edges = np.linspace(10.0, 330.0, 33)
    rebin, rebin_edges = rebin_hist(hist, 4, bin_edges, 20.0, 300.0)
    assert rebin.shape == (7,) * ndim
    assert np.isclose(hist[(slice(1, -3),) * ndim].sum(), rebin.sum() * 4)
    assert np.isclose(hist[(slice(1, 5),) * ndim].sum(), rebin[(0,) * ndim] * 4)
    assert np.isclose(rebin_edges, np.linspace(20.0, 300.0, 8)).all()


def test_rebin_hist_bin_edges_lr():
    hist = np.random.uniform(size=(32, 32))
    bin_edges = np.linspace(10.0, 330.0, 33)
    rebin, rebin_edges = rebin_hist(hist, 4, bin_edges, 25.0, 270.0)
    assert rebin.shape == (7, 7)
    assert np.isclose(rebin_edges, np.linspace(20.0, 300.0, 8)).all()


def test_read_spectrum_no_bin_edges_npz(tmp_path):
    path = tmp_path / "test.npz"
    hist = np.linspace(0.0, 10.0, 11)
    np.savez_compressed(path, **{"test": hist})

    res, bin_edges = read_spectrum(path, "test", False)
    assert not bin_edges
    assert np.isclose(res, hist).all()

    with pytest.raises(ValueError):
        read_spectrum(path, "test", True)


@pytest.mark.skipif(
    find_spec("h5py") is None, reason="Module h5py not installed"
)
def test_read_spectrum_no_bin_edges_hdf5(tmp_path):
    import h5py

    path = tmp_path / "test.hdf5"
    hist = np.linspace(0.0, 10.0, 11)
    with h5py.File(path, "w") as f:
        f.create_dataset("test", data=hist)

    res, bin_edges = read_spectrum(path, "test", False)
    assert not bin_edges
    assert np.isclose(res, hist).all()

    with pytest.raises(ValueError):
        read_spectrum(path, "test", True)


def test_read_pos_int_spectrum(tmp_path):
    bin_edges = np.linspace(0.0, 10.0, 11)

    hist = np.linspace(1.0, 10.0, 10, dtype=float)
    write_hist(tmp_path / "test_float.npz", "test", hist, bin_edges)
    res_hist, (res_bin_edges,) = read_pos_int_spectrum(
        tmp_path / "test_float.npz", "test"
    )
    assert np.isclose(res_bin_edges, bin_edges).all()
    assert np.isclose(res_hist, hist).all()
    assert issubclass(res_hist.dtype.type, np.integer)

    hist = np.linspace(0.5, 17.0, 10, dtype=float)
    write_hist(tmp_path / "test_nonint.npz", "test", hist, bin_edges)
    with pytest.raises(ValueError):
        read_pos_int_spectrum(tmp_path / "test_nonint.npz", "test")

    hist = np.linspace(-10, -1.0, 10, dtype=float)
    write_hist(tmp_path / "test_neg.npz", "test", hist, bin_edges)
    with pytest.raises(ValueError):
        read_pos_int_spectrum(tmp_path / "test_neg.npz", "test")


def test_read_rebin_spectrum(tmp_path):
    bin_edges = np.linspace(0.0, 10.0, 11)
    hist = np.linspace(1.0, 10.0, 10, dtype=float)
    path = tmp_path / "test.npz"
    write_hist(path, "test", hist, bin_edges)

    target_bin_edges = np.linspace(-2.0, 12.0, 8)
    res_hist, (res_bin_edges,) = read_rebin_spectrum(
        path, target_bin_edges, "test"
    )
    assert res_hist.sum() == hist.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(res_bin_edges, bin_edges).all()

    read_rebin_spectrum(path, target_bin_edges, "test", [0.0, 1.0])

    path1 = tmp_path / "test1.npz"
    np.savez_compressed(path1, **{"test": hist})
    res_hist, (res_bin_edges,) = read_rebin_spectrum(
        path1, target_bin_edges, "test", cal_bin_edges=[0.0, 1.0]
    )
    assert res_hist.sum() == hist.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(res_bin_edges, bin_edges).all()

    res_hist, (res_bin_edges,) = read_rebin_spectrum(
        path1, target_bin_edges, "test", cal_bin_centers=[0.5, 1.0]
    )
    assert res_hist.sum() == hist.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(res_bin_edges, bin_edges).all()

    path2 = tmp_path / "test.root"
    write_hist(path2, "test", hist, bin_edges)
    res_hist, (res_bin_edges,) = read_rebin_spectrum(
        path2, target_bin_edges, "test", cal_bin_centers=[0.5, 1.0]
    )
    assert res_hist.sum() == hist.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(res_bin_edges, bin_edges).all()


def test_read_rebin_spectrum_filter(tmp_path):
    bin_edges = np.linspace(0.0, 10.0, 11)
    hist = np.linspace(1.0, 10.0, 10, dtype=float)
    path = tmp_path / "test.npz"
    write_hist(path, "test", hist, bin_edges)

    target_bin_edges = np.linspace(-2.0, 12.0, 8)
    res_hist, _ = read_rebin_spectrum(
        path, target_bin_edges, "test", filter_spectrum=lambda x: x * 2
    )
    assert res_hist.sum() == 2.0 * hist.sum()


def test_get_rema(tmp_path):
    rema = np.ones((10, 10))
    hist_norm = 100 * np.ones(10)
    bin_edges = np.linspace(0, 10, 11)
    path = tmp_path / "rema.npz"
    write_hists(
        {"rema": rema, "n_simulated_particles": hist_norm},
        [bin_edges, bin_edges],
        path,
    )

    res_rema, res_bin_edges = get_rema(path, "rema", 2, 0, 10)
    assert res_rema.shape == (5, 5)
    assert np.isclose(res_rema.sum() * 2, rema.sum())
    assert res_bin_edges.shape == (6,)

    res_rema, res_bin_edges = get_rema(
        path, "rema", 2, 0, 10, "n_simulated_particles"
    )
    assert res_rema.shape == (5, 5)
    assert np.isclose(100.0 * res_rema.sum() * 2, rema.sum())
    assert res_bin_edges.shape == (6,)

    rema = np.ones((10, 20))
    path1 = tmp_path / "rema1.npz"
    write_hist(path1, "rema", rema, [bin_edges, bin_edges])
    with pytest.raises(ValueError):
        get_rema(path1, "rema", 2, 0, 10)


def test_get_rema_unequal_binning(tmp_path):
    import uproot

    rema = (np.ones((10, 10)), np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    hist_norm = (np.ones(20), np.linspace(0, 20, 21))
    path = tmp_path / "rema.root"
    with uproot.recreate(path) as f:
        f["rema"] = rema
        f["n_simulated_particles"] = hist_norm

    with pytest.raises(ValueError):
        get_rema(path, "rema", 2, 0, 10, "n_simulated_particles")


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


def test_create_matrix(tmp_path):
    simulations = create_simulations(tmp_path)

    mat, bin_edges, bin_edges2 = create_matrix(
        simulations, "det1", scale_hist_axis=1.0
    )
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 600
    assert np.isclose(bin_edges[0], 0.0)
    assert np.isclose(bin_edges[-1], 600.0)

    mat, bin_edges, bin_edges2 = create_matrix(
        simulations, "det1", scale_hist_axis=1.0, max_energy=1000.0
    )
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 1000
    assert np.isclose(mat.diagonal(), np.ones(1000) * 0.05).all()


def test_create_matrix_tv(tmp_path):
    simulations = create_simulations(
        tmp_path, detectors=["det2"], shift_axis=-0.5
    )
    mat, bin_edges, bin_edges2 = create_matrix(
        simulations, "det2", scale_hist_axis=1.0
    )
    assert (bin_edges == bin_edges2).all()
    assert mat.shape[0] == mat.shape[1] == bin_edges.shape[0] - 1 == 601
    assert np.isclose(bin_edges[0], -0.5)
    assert np.isclose(bin_edges[-1], 600.5)
