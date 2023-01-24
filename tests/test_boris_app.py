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

import sys
from unittest import mock

import numpy as np
import hist

from boris.io import write_specs
from boris.boris_app import BorisApp, init


def test_BorisApp(tmp_path):
    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        "--fit-beam",
        "--ndraws=50",
        "--bg-spectrum",
        str(tmp_path / "background.npz"),
        "--matrixfile-alt",
        str(tmp_path / "rema_alt.npz"),
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed.npz"),
        str(tmp_path / "incident.nc"),
    ]

    rema = (
        hist.Hist.new.Regular(10, 2000, 2200)
        .Regular(10, 2000, 2200)
        .Double(data=0.1 * np.diag(np.ones(10)) + 0.01 * np.diag(np.ones(8), 2))
    )
    rema_alt = (
        hist.Hist.new.Regular(10, 2000, 2200)
        .Regular(10, 2000, 2200)
        .Double(data=0.1 * np.diag(np.ones(10)) + 0.02 * np.diag(np.ones(8), 2))
    )
    incident = hist.Hist.new.Regular(10, 2000, 2200).Int64(
        data=np.random.uniform(10, 1000, size=10).astype(np.int64)
    )
    background = hist.Hist.new.Regular(10, 2000, 2200).Int64(
        data=np.random.uniform(10, 100, size=10).astype(np.int64)
    )
    observed_wobg = hist.Hist.new.Regular(10, 2000, 2200).Int64(
        data=(incident.values() @ rema.values()).astype(np.int64)
    )
    observed = hist.Hist.new.Regular(10, 2000, 2200).Int64(
        data=(background.values() + incident.values() @ rema.values()).astype(
            np.int64
        )
    )

    write_specs(tmp_path / "rema.npz", {"rema": rema})
    write_specs(tmp_path / "rema_alt.npz", {"rema": rema_alt})
    write_specs(tmp_path / "observed.npz", {"observed": observed})
    write_specs(tmp_path / "observed_wobg.npz", {"observed": observed_wobg})
    write_specs(tmp_path / "background.npz", {"background": background})

    BorisApp()
    (tmp_path / "incident.npz").exists()

    sys.argv = [
        "boris",
        "--seed=0",
        "--left=2000",
        "--right=2200",
        "--binning-factor=1",
        "--tune=50",
        str(tmp_path / "rema.npz"),
        str(tmp_path / "observed_wobg.npz"),
        str(tmp_path / "incident_wobg.npz"),
    ]
    BorisApp()
    assert (tmp_path / "incident_wobg.npz").exists()


@mock.patch("boris.boris_app.BorisApp")
@mock.patch("boris.boris_app.__name__", "__main__")
def test_app_init_BorisApp(app):
    init()
    assert app.called
