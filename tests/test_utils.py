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

import numpy as np

from boris.utils import _bin_edges_dict, hdi, get_obj_by_name, get_obj_bin_edges


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


@pytest.mark.skip
def test_numpy_to_root_hist():
    ...


@pytest.mark.skip
def test_write_hist():
    ...


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
