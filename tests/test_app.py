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
import hist

import boris.app
from boris.boris_app import boris_app
from boris.sirob_app import sirob_app
from boris.boris2spec_app import boris2spec_app
from boris.checkmatrix_app import checkmatrix_app
from boris.makematrix_app import makematrix_app
from boris.app import (
    do_step,
    logger,
    check_if_exists,
    make_matrix,
    setup_logging,
    sirob,
)
from boris.io import write_specs, read_spectrum

from tests.helpers.utils import create_simulations


@pytest.fixture
def rema():
    return (
        hist.Hist.new.Regular(10, 2000.0, 2200.0)
        .Regular(10, 2000.0, 2200.0)
        .Double(data=0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2))
    )


@pytest.fixture
def incident():
    return hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=np.random.uniform(10, 1000, size=10).astype(np.int64)
    )


@pytest.mark.parametrize(
    "app, name",
    [
        (boris_app, "boris"),
        (checkmatrix_app, "checkmatrix"),
        (sirob_app, "sirob"),
        (boris2spec_app, "boris2spec"),
        (makematrix_app, "makematrix"),
    ],
)
def test_help(app, name):
    sys.argv = [name, "--help"]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        app()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_sirob(rema, incident, tmp_path):
    write_specs(tmp_path / "rema.npz", {"rema": rema})
    write_specs(tmp_path / "incident.npz", {"incident": incident})
    sirob(
        tmp_path / "rema.npz",
        tmp_path / "incident.npz",
        tmp_path / "observed.npz",
        1,
        2000,
        2200,
    )
    assert (tmp_path / "observed.npz").exists()
    observed = read_spectrum(tmp_path / "observed.npz")
    assert observed.ndim == 1
    assert observed.shape[0] == 10
    assert np.isclose(observed.axes[0].edges, incident.axes[0].edges).all()
    assert np.isclose(
        incident.values() @ rema.values(), observed.values()
    ).all()

    background = hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=np.random.uniform(10, 100, size=10).astype(np.int64)
    )
    write_specs(tmp_path / "background.npz", {"background": background})

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
    observed_bg = read_spectrum(tmp_path / "observed_bg.npz")
    assert observed_bg.ndim == 1
    assert observed_bg.shape[0] == 10
    assert np.isclose(observed_bg.axes[0].edges, incident.axes[0].edges).all()
    assert np.isclose(
        incident.values() @ rema.values() + 2.0 * background.values(),
        observed_bg.values(),
    ).all()


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


def test_boris(rema, incident, tmp_path):
    observed = hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=(incident.values() @ rema.values()).astype(np.int64)
    )

    write_specs(tmp_path / "rema.npz", {"rema": rema})
    write_specs(tmp_path / "observed.npz", {"observed": observed})

    boris.app.boris(
        tmp_path / "rema.npz",
        tmp_path / "observed.npz",
        tmp_path / "incident.nc",
        1,
        2000,
        2200,
        ndraws=10,
        tune=10,
        cores=1,
    )
    assert (tmp_path / "incident.nc").exists()

    background = hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=np.random.uniform(10, 100, size=10).astype(np.int64)
    )
    observed_bg = hist.Hist.new.Regular(10, 2000.0, 2200.0).Int64(
        data=(incident.values() @ rema.values() + background.values())
    )
    write_specs(tmp_path / "observed_bg.npz", {"observed_bg": observed_bg})
    write_specs(tmp_path / "background.npz", {"background": background})

    boris.app.boris(
        tmp_path / "rema.npz",
        tmp_path / "observed_bg.npz",
        tmp_path / "incident_bg.nc",
        10,
        2000,
        2200,
        ndraws=1000,
        tune=200,
        cores=1,
        background_spectrum=tmp_path / "background.npz",
    )
    assert (tmp_path / "incident_bg.nc").exists()

    boris.app.boris2spec(
        tmp_path / "incident_bg.nc",
        tmp_path / "output.root",
        # plot="plot.png",
        # plot_title="title",
        # plot_xlabel="xlabel",
        # plot_ylabel="ylabel",
        get_mean=True,
        get_median=True,
        get_variance=True,
        get_std_dev=True,
        get_min=True,
        get_max=True,
        get_hdi=True,
    )
    assert (tmp_path / "output.root").exists()


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
    )
    assert (tmp_path / filename).exists()
    for det in detectors:
        mat = read_spectrum(tmp_path / filename, det)
        assert (mat.axes[0].edges == mat.axes[1].edges).all()
        assert np.isclose(mat.axes[0].edges[0], 0.0)
        assert np.isclose(mat.axes[0].edges[-1], 600.0)


def test_check_if_exists(tmp_path):
    with open(tmp_path / "test.txt", "w") as f:
        print("", file=f)
    with pytest.raises(Exception):
        check_if_exists(tmp_path / "test.txt")
