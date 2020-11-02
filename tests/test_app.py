#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2020 O. Papst.
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

from boris.app import BorisApp, SirobApp, Boris2SpecApp


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
