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

import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def temp_file(request, *args, **kwargs):
    """
    pytest fixture that provides a temporary file for writing
    """
    filename = tempfile.mkstemp(prefix="boris_", *args, **kwargs)[1]
    os.remove(filename)
    yield Path(filename)
    try:
        os.remove(filename)
    except OSError:
        pass


@pytest.fixture(
    scope="function",
    params=[
        ".npz",
        pytest.param(
            ".hdf5",
            marks=pytest.mark.skipif(
                "h5py" not in sys.modules, reason="Module h5py not installed"
            ),
        ),
        pytest.param(
            ".root",
            marks=pytest.mark.skipif(
                ("uproot3" not in sys.modules or "uproot" not in sys.modules),
                reason="Module uproot and/or uproot3 not installed",
            ),
        ),
        ".txt",
    ],
)
def container_file(request):
    """
    pytest fixture that provides a temporary container for writing
    """
    yield from temp_file(request, suffix=str(request.param))
