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

from boris.makematrix_app import makematrix_app

from tests.helpers.utils import create_simulations


def test_makematrix_app(tmp_path):
    simulations = create_simulations(tmp_path)

    path = tmp_path / "sim.dat"
    with open(path, "w") as f:
        for sim in simulations:
            print(str(sim), file=f, end="\n")

    sys.argv = [
        "makematrix",
        "--detector",
        "det1",
        "--",
        str(path),
        str(tmp_path / "test.root"),
    ]
    makematrix_app()
