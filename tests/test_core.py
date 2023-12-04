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

from boris.core import fit


def test_fit_mismatch():
    bin_edges = np.linspace(0, 10, 11)
    rema = np.random.uniform((10, 10))
    spectrum = np.random.poisson(1000, size=10)
    background = np.random.poisson(1000, size=12)
    with pytest.raises(ValueError):
        fit(rema, spectrum.T, bin_edges, background.T)
