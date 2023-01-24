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

import numpy as np
import hist

from boris.io import (
    SimInfo,
    obj_resolve_name,
    get_simulation_spectra,
    write_specs,
)
from boris.utils import (
    create_matrix,
    get_bin_edges_from_calibration,
    get_rema,
    interpolate_grid,
    read_rebin_spectrum,
    rebin_uniform,
)


def test_get_bin_edges_from_calibration():
    cal = np.array([0.0, 2.0])

    res = get_bin_edges_from_calibration(10, cal)
    assert res.shape[0] == 11
    assert len(res.shape) == 1
    assert type(res) == np.ndarray
    assert res.dtype == float
    assert res[0] == cal[0]
    assert res[1] == cal[0] + cal[1]

    res_edges = get_bin_edges_from_calibration(10, cal, "edges")
    assert (res == res_edges).all()

    res_centers = get_bin_edges_from_calibration(10, cal, "centers")
    assert (res != res_centers).all()
    assert res_centers[1] == cal[0] + cal[1] * 0.5

    with pytest.raises(KeyError):
        get_bin_edges_from_calibration(10, cal, "centroid")


def test_obj_resolve_name_single():
    res = obj_resolve_name({"a": 1})
    assert res == "a"


def test_obj_resolve_name_single_ignore_bin_edges():
    res = obj_resolve_name({"a": 1, "edges": 2})
    assert res == "a"


def test_obj_resolve_name_name():
    res = obj_resolve_name({"a": 1}, "a")
    assert res == "a"


def test_obj_resolve_name_not_found():
    with pytest.raises(KeyError):
        obj_resolve_name({})

    with pytest.raises(KeyError):
        obj_resolve_name({"a": 1, "b": 2})

    with pytest.raises(KeyError):
        obj_resolve_name({"a": 1, "b": 2, "edges": 2})


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


def test_read_rebin_spectrum(tmp_path):
    h = hist.Hist.new.Regular(10, 0, 10).Int64(
        data=np.random.uniform(0, 100, size=10)
    )
    path = tmp_path / "test.npz"
    write_specs(path, {"test": h})

    target_bin_edges = np.linspace(-2.0, 12.0, 8)
    res_hist = read_rebin_spectrum(path, target_bin_edges, "test")
    assert res_hist.sum() == h.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(target_bin_edges, res_hist.axes[0].edges).all()

    read_rebin_spectrum(path, target_bin_edges, "test", [0.0, 1.0])

    path1 = tmp_path / "test1.npz"
    np.savez_compressed(path1, **{"test": h, "test_edges": h.axes[0].edges})
    res_hist = read_rebin_spectrum(
        path1, target_bin_edges, "test", calibration=[0.0, 1.0]
    )
    assert res_hist.sum() == h.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(target_bin_edges, res_hist.axes[0].edges).all()

    res_hist = read_rebin_spectrum(
        path1,
        target_bin_edges,
        "test",
        calibration=[0.5, 1.0],
        convention="centers",
    )
    assert res_hist.sum() == h.sum()
    assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    assert np.isclose(target_bin_edges, res_hist.axes[0].edges).all()

    h2 = hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=np.random.uniform(10, 100, size=10).astype(np.int64)
    )
    write_specs(tmp_path / "test2.npz", {"somename": h2})
    test = read_rebin_spectrum(
        tmp_path / "test2.npz", np.linspace(2000.0, 2200.0, 11)
    )
    assert np.isclose(test.axes[0].edges, np.linspace(2000.0, 2200.0, 11)).all()
    print(h2.sum())
    print(test.sum())
    assert test.sum() == h2.sum()

    test = read_rebin_spectrum(
        tmp_path / "test2.npz", np.linspace(2000.0, 2200.0, 101)
    )
    assert np.isclose(
        test.axes[0].edges, np.linspace(2000.0, 2200.0, 101)
    ).all()
    assert test.sum() == h2.sum()
    print(test.sum())

    # TODO: Bug in numpy: https://github.com/numpy/numpy/issues/12264
    # path2 = tmp_path / "test.root"
    # write_specs(path2, {"test": h})
    # res_hist = read_rebin_spectrum(
    #    path2, target_bin_edges, "test", calibration=[0.5, 1.0], convention="centers",
    # )
    # assert res_hist.sum() == h.sum()
    # assert res_hist.shape[0] == target_bin_edges.shape[0] - 1
    # assert np.isclose(target_bin_edges, res_hist.axes[0].edges).all()


def test_read_rebin_spectrum_negative(tmp_path):
    h = hist.Hist.new.Regular(10, 0, 10).Int64(
        data=np.random.uniform(-100, 100, size=10)
    )
    path = tmp_path / "test.npz"
    write_specs(path, {"test": h})

    target_bin_edges = np.linspace(-2.0, 12.0, 8)
    with pytest.raises(ValueError):
        read_rebin_spectrum(path, target_bin_edges, "test")


def test_read_rebin_spectrum_non_int(tmp_path):
    h = hist.Hist.new.Regular(10, 0, 10).Double(
        data=np.random.uniform(0, 100, size=10)
    )
    path = tmp_path / "test.npz"
    write_specs(path, {"test": h})

    target_bin_edges = np.linspace(-2.0, 12.0, 8)
    with pytest.raises(ValueError):
        read_rebin_spectrum(path, target_bin_edges, "test")


def test_get_rema(tmp_path):
    rema = (
        hist.Hist.new.Regular(10, 0, 100)
        .Regular(10, 0, 100)
        .Double(data=np.ones((10, 10)))
    )
    path = tmp_path / "rema.npz"
    write_specs(path, {"rema": rema})

    res_rema = get_rema(path, "rema", 2, 0, 100)
    assert res_rema.shape == (5, 5)
    assert np.isclose(res_rema.sum(), rema.sum())

    rema = (
        hist.Hist.new.Regular(10, 0, 100)
        .Regular(20, 0, 100)
        .Double(data=np.ones((10, 20)))
    )
    path1 = tmp_path / "rema1.npz"
    write_specs(path1, {"rema": rema})
    with pytest.raises(ValueError):
        get_rema(path1, "rema", 2, 0, 10)


def test_SimInfo_from_line():
    res = SimInfo.from_dat_file_line("test.root: 1332.0 1000", Path("/"))

    assert res.path == Path("/test.root")
    assert res.energy == 1332.0
    assert res.nevents == 1000


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
    sim_spectra = get_simulation_spectra(simulations, "det1")

    mat = create_matrix(sim_spectra, 600, 0, 600)
    assert (mat.axes[0].edges == mat.axes[0].edges).all()
    assert mat.shape[0] == mat.shape[1] == 600
    assert np.isclose(mat.axes[0].edges[0], 0.0)
    assert np.isclose(mat.axes[0].edges[-1], 600.0)

    mat = create_matrix(sim_spectra, 1000, 0, 1000)
    assert (mat.axes[0].edges == mat.axes[0].edges).all()
    assert mat.shape[0] == mat.shape[1] == 1000
    assert np.isclose(mat.values().diagonal(), np.ones(1000) * 0.05).all()


def test_create_matrix_tv(tmp_path):
    simulations = create_simulations(
        tmp_path, detectors=["det2"], shift_axis=-0.5
    )
    sim_spectra = get_simulation_spectra(simulations, "det2")
    mat = create_matrix(sim_spectra, 601, -0.5, 600.5)
    assert (mat.axes[0].edges == mat.axes[1].edges).all()
    assert mat.shape[0] == mat.shape[1] == 601
    assert np.isclose(mat.axes[0].edges[0], -0.5)
    assert np.isclose(mat.axes[0].edges[-1], 600.5)
