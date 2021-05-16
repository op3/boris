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
from unittest import mock

import numpy as np

import boris.app
from boris.app import (
    BorisApp,
    SirobApp,
    Boris2SpecApp,
    sirob,
    setup_logging,
    logger,
    do_step,
    init,
)
from boris.utils import write_hist, read_spectrum


@pytest.mark.parametrize(
    "app, name",
    [
        (BorisApp, "boris"),
        (SirobApp, "sirob"),
        (Boris2SpecApp, "boris2spec"),
    ],
)
def test_help(app, name):
    sys.argv = ["boris", "--help"]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        app()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_BorisApp(tmp_path):
    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        "--burn=50",
        "--bg-spectrum",
        str(tmp_path / "background.npz"),
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed.npz"),
        str(tmp_path / "incident.npz"),
    ]
    rema = 0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2)
    bin_edges = np.linspace(2000, 2200, 11)
    incident = np.random.uniform(10, 1000, size=10).astype(np.int64)
    background = np.random.uniform(10, 100, size=10).astype(np.int64)
    observed = (incident @ rema).astype(np.int64)
    observed = (incident @ rema + background).astype(np.int64)
    write_hist(tmp_path / "rema.npz", "rema", rema, bin_edges)
    write_hist(tmp_path / "observed.npz", "observed", observed, bin_edges)
    write_hist(tmp_path / "background.npz", "background", background, bin_edges)
    BorisApp()
    assert (tmp_path / "incident.npz").exists()

    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        "--burn=50",
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed.npz"),
        str(tmp_path / "incident_wobg.npz"),
    ]
    BorisApp()
    assert (tmp_path / "incident_wobg.npz").exists()


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

    def _deconvolute(*args, **kwargs):
        assert kwargs["ndraws"] == 1
        assert kwargs["tune"] == 1
        assert kwargs["thin"] == 1
        assert kwargs["burn"] == 1
        return {"incident": np.ones((10, 10))}

    boris.app.boris(
        tmp_path / "rema.npz",
        tmp_path / "observed.npz",
        tmp_path / "incident.npz",
        1,
        2000,
        2200,
        ndraws=1,
        tune=1,
        thin=1,
        burn=1,
        cores=1,
        deconvolute=_deconvolute,
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
        ndraws=1,
        tune=1,
        thin=1,
        burn=1,
        cores=1,
        background_spectrum=tmp_path / "background.npz",
        deconvolute=_deconvolute,
    )
    assert (tmp_path / "incident_bg.npz").exists()


def test_Boris2SpecApp(tmp_path):
    incident = np.ones((100, 10))
    bin_edges = np.linspace(2000, 2200, 11)
    write_hist(tmp_path / "incident.npz", "incident", incident, bin_edges)

    sys.argv = [
        "boris2spec",
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
    Boris2SpecApp()
    assert (tmp_path / "output.npz").exists()
    mean, (bin_edges,) = read_spectrum(tmp_path / "output.npz", "mean")
    assert np.isclose(mean, np.ones(10)).all()
    assert bin_edges.shape[0] == 11

    for key in ["mean", "median", "var", "std", "hdi_lo", "hdi_hi"]:
        res, (bin_edges,) = read_spectrum(tmp_path / "output.npz", key)
        assert res.ndim == 1
        assert res.shape[0] == 10


@mock.patch.object(boris.app, "sirob")
def test_SirobApp(mock_sirob):
    sys.argv = ["sirob", "rema.npz", "incident.npz", "observed.npz"]
    SirobApp()
    assert mock_sirob.called


@mock.patch("matplotlib.pyplot")
def test_Boris2SpecApp_plot(mock_plt, tmp_path):
    incident = np.ones((100, 10))
    bin_edges = np.linspace(2000, 2200, 11)
    write_hist(tmp_path / "incident.npz", "incident", incident, bin_edges)

    sys.argv = [
        "boris2spec",
        "--get-mean",
        "--get-hdi",
        "--plot",
        str(tmp_path / "incident.npz"),
    ]
    Boris2SpecApp()
    assert mock_plt.legend.called
    assert mock_plt.show.called


def test_Boris2SpecApp_wrong_args(tmp_path):
    sys.argv = ["boris2spec", str(tmp_path / "incident.npz")]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Boris2SpecApp()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2

    sys.argv = [
        "boris2spec",
        str(tmp_path / "incident.npz"),
        str(tmp_path / "output.npz"),
    ]
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        Boris2SpecApp()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


@mock.patch("boris.app.SirobApp")
@mock.patch("boris.app.__name__", "__main__")
def test_app_init_SirobApp(app):
    sys.argv = ["sirob"]
    init()
    assert app.called


@mock.patch("boris.app.BorisApp")
@mock.patch("boris.app.__name__", "__main__")
def test_app_init_BorisApp(app):
    sys.argv = ["boris"]
    init()
    assert app.called


@mock.patch("boris.app.Boris2SpecApp")
@mock.patch("boris.app.__name__", "__main__")
def test_app_init_Boris2SpecApp(app):
    sys.argv = ["boris2spec"]
    init()
    assert app.called
