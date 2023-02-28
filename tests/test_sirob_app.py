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

import boris.app
from boris.sirob_app import sirob_app


@mock.patch.object(boris.app, "sirob")
def test_SirobApp(mock_sirob):
    sys.argv = ["sirob", "rema.npz", "incident.npz", "observed.npz"]
    sirob_app()
    assert mock_sirob.called
