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
import logging
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import arviz

import boris.app
from boris.boris_app import BorisApp
from boris.sirob_app import SirobApp
from boris.boris2spec_app import Boris2SpecApp
from boris.checkmatrix_app import CheckMatrixApp
from boris.makematrix_app import MakeMatrixApp
from boris.app import (
    do_step,
    logger,
    check_if_exists,
    make_matrix,
    setup_logging,
    sirob,
)
from boris.utils import write_hist, read_spectrum

from tests.helpers.utils import create_simulations


@pytest.mark.parametrize(
    "app, name",
    [
        (BorisApp, "boris"),
        (CheckMatrixApp, "checkmatrix"),
        (SirobApp, "sirob"),
        (Boris2SpecApp, "boris2spec"),
        (MakeMatrixApp, "makematrix"),
    ],
)
def test_help(app, name):
    sys.argv = [name, "--help"]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        app()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_sirob(tmp_path):
    rema = 0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2)
    bin_edges = np.linspace(2000, 2200, 11)
    incident = np.random.uniform(10, 1000, size=10).astype(np.int64)
    write_hist(tmp_path / "rema.npz", "rema", rema, bin_edges)
    write_hist(tmp_path / "incident.npz", "incident", incident, bin_edges)
    sirob(
        tmp_path / "rema.npz",
        tmp_path / "incident.npz",
        tmp_path / "observed.npz",
        1,
        2000,
        2200,
    )
    assert (tmp_path / "observed.npz").exists()
    observed, (obs_bin_edges,) = read_spectrum(tmp_path / "observed.npz")
    assert observed.ndim == 1
    assert observed.shape[0] == 10
    assert np.isclose(obs_bin_edges, bin_edges).all()
    assert np.isclose(incident @ rema, observed).all()

    background = np.random.uniform(10, 100, size=10).astype(np.int64)
    write_hist(tmp_path / "background.npz", "background", background, bin_edges)
    sirob(
        tmp_path / "rema.npz",
        tmp_path / "incident.npz",
        tmp_path / "observed_bg.npz",
        1,
        2000,
        2200,
        background_spectrum=(tmp_path / "background.npz"),
        background_name="background",
        background_scale=2.0,
    )
    assert (tmp_path / "observed_bg.npz").exists()
    observed, (obs_bin_edges,) = read_spectrum(tmp_path / "observed_bg.npz")
    assert observed.ndim == 1
    assert observed.shape[0] == 10
    assert np.isclose(obs_bin_edges, bin_edges).all()
    assert np.isclose(incident @ rema + 2.0 * background, observed).all()


def test_setup_logging():
    before = len(logger.handlers)
    setup_logging()
    assert logger.level == logging.INFO
    assert before + 1 == len(logger.handlers)


def test_do_step():
    with do_step("some text"):
        pass

    with do_step("simple text", True):
        pass

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        with do_step("this fails"):
            raise Exception("oops")
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_boris(tmp_path):
    rema = 0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2)
    bin_edges = np.linspace(2000, 2200, 11)
    incident = np.random.uniform(10, 1000, size=10).astype(np.int64)
    observed = (incident @ rema).astype(np.int64)
    write_hist(tmp_path / "rema.npz", "rema", rema, bin_edges)
    write_hist(tmp_path / "observed.npz", "observed", observed, bin_edges)

    boris.app.boris(
        tmp_path / "rema.npz",
        tmp_path / "observed.npz",
        tmp_path / "incident.npz",
        1,
        2000,
        2200,
        ndraws=10,
        tune=10,
        thin=1,
        burn=10,
        cores=1,
    )
    assert (tmp_path / "incident.npz").exists()

    background = np.random.uniform(10, 100, size=10).astype(np.int64)
    observed_bg = (incident @ rema + background).astype(np.int64)
    write_hist(
        tmp_path / "observed_bg.npz", "observed_bg", observed_bg, bin_edges
    )
    write_hist(tmp_path / "background.npz", "background", background, bin_edges)

    boris.app.boris(
        tmp_path / "rema.npz",
        tmp_path / "observed_bg.npz",
        tmp_path / "incident_bg.npz",
        1,
        2000,
        2200,
        ndraws=10,
        tune=10,
        thin=1,
        burn=10,
        cores=1,
        background_spectrum=tmp_path / "background.npz",
    )
    assert (tmp_path / "incident_bg.npz").exists()


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
            print(str(sim), file=f, end="\n")

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


def test_check_if_exists(tmp_path):
    with open(tmp_path / "test.txt", "w") as f:
        print("", file=f)
    with pytest.raises(Exception):
        check_if_exists(tmp_path / "test.txt")
