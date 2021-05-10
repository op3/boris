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

from tests.helpers.utils import hide_module

from importlib.util import find_spec

import numpy as np

from boris.utils import (
    rebin_hist,
    rebin_uniform,
    load_rema,
    get_rema,
    _bin_edges_dict,
    hdi,
    get_obj_by_name,
    get_obj_bin_edges,
    numpy_to_root_hist,
    write_hist,
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


pytest.mark.skipif(find_spec("uproot3") is None, reason="uproot3 not installed")


def test_numpy_to_root_hist_1D():
    import uproot3 as uproot

    hist = np.random.uniform(size=10)
    bin_edges = np.linspace(0, 10, 11)
    res = numpy_to_root_hist(hist, bin_edges)
    assert res._classname == b"TH1D"
    assert np.isclose(res._fEntries, hist.sum())
    assert res._fXaxis._fNbins == bin_edges.shape[0] - 1


pytest.mark.skipif(find_spec("uproot3") is None, reason="uproot3 not installed")


def test_numpy_to_root_hist_2D():
    import uproot3 as uproot

    hist = np.random.uniform(size=(10, 20))
    bin_edges = [np.linspace(0, 10, 11), np.linspace(0, 20, 21)]
    res = numpy_to_root_hist(hist, bin_edges)
    assert res._classname == b"TH2D"
    assert np.isclose(res._fEntries, hist.sum())
    assert res._fXaxis._fNbins == bin_edges[0].shape[0] - 1
    assert res._fYaxis._fNbins == bin_edges[1].shape[0] - 1


pytest.mark.skipif(find_spec("uproot3") is None, reason="uproot3 not installed")


def test_numpy_to_root_hist_2D_trace():
    import uproot3 as uproot

    hist = np.random.uniform(size=(100, 10))
    bin_edges = np.linspace(0, 10, 11)
    res = numpy_to_root_hist(hist, bin_edges)
    assert res._classname == b"TH2D"
    assert np.isclose(res._fEntries, hist.sum())
    assert res._fXaxis._fNbins == 100
    assert res._fYaxis._fNbins == bin_edges.shape[0] - 1


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


def test_write_hist_exists(tmp_path):
    hist = np.random.uniform(size=100)
    bin_edges = np.linspace(0, 100, 101)
    with pytest.raises(Exception):
        write_hist(tmp_path, "test", hist, bin_edges)


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


def test_write_hist_without_h5py(tmp_path):
    hist = np.random.uniform(size=(10, 10))
    bin_edges = np.linspace(0, 10, 11)
    with hide_module("h5py"):
        with pytest.raises(ModuleNotFoundError):
            write_hist(tmp_path / "test.hdf5", "test", hist, bin_edges)


@pytest.mark.skip
def test_write_hists():
    ...


@pytest.mark.skip
def test_get_filetype():
    ...


@pytest.mark.skip
def test_get_bin_edges():
    ...


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
    assert type(res) == list
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


@pytest.mark.skip
def test_get_obj_bin_edges():
    ...


@pytest.mark.skip
def test_get_keys_in_container():
    ...


@pytest.mark.skip
def test_read_spectrum():
    ...


@pytest.mark.skip
def test_read_pos_int_spectrum():
    ...


@pytest.mark.skip
def test_read_rebin_spectrum():
    ...


def test_hdi():
    sample = np.linspace(-10, 10, 1001)
    sample = sample ** 3
    lo, hi = hdi(sample)
    assert lo == sample[158]
    assert hi == sample[841]


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
    assert np.isclose(hist.sum(), rebin.sum())
    for i, j in zip(hist.shape, rebin.shape):
        assert i == 4 * j


@pytest.mark.parametrize(
    "bin_width,res_bin_edges",
    [
        (2, np.linspace(6.0, 26.0, 11)),
        (3, np.linspace(6.0, 24.0, 7)),
    ],
)
def test_rebin_hist_lr(bin_width, res_bin_edges):
    hist = np.random.uniform(size=(32, 32))
    rebin, bin_edges = rebin_hist(hist, bin_width, left=6, right=26)
    assert (np.isclose(bin_edges, res_bin_edges)).all()
    assert rebin.shape[0] + 1 == bin_edges.shape[0]
    assert rebin.shape[1] + 1 == bin_edges.shape[0]


def test_rebin_hist_nonsquare():
    hist = np.random.uniform(size=(32, 16))
    with pytest.raises(ValueError):
        rebin_hist(hist, 2)
