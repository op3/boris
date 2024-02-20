#!/usr/bin/env python
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

"""checkmatrix cli for detector response visualisation."""

from __future__ import annotations

import argparse
from pathlib import Path


def checkmatrix_app():
    """CLI interface for boris2spec."""
    parser = argparse.ArgumentParser(description="Display detector response matrix.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity",
        action="store_true",
    )
    parser.add_argument(
        "matrixfile",
        help="container file containing detector response matrix",
        type=Path,
    )
    parser.add_argument(
        "-l",
        "--left",
        help="crop response matrix to the lowest bin still containing LEFT (default: %(default)s)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-r",
        "--right",
        help="crop response matrix to the highest bin not containing RIGHT (default: maximum energy of simulation)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-b",
        "--binning-factor",
        help="rebinning factor, group this many bins together (default: %(default)s)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--rema-name",
        help="name of the detector response matrix in matrix file  (default: %(default)s)",
        default="rema",
        nargs="?",
        type=str,
    )
    args = parser.parse_args()

    from boris.app import setup_logging, check_matrix

    setup_logging(args.verbose)
    check_matrix(
        args.matrixfile,
        args.binning_factor,
        args.left,
        args.right,
        args.rema_name,
    )


if __name__ == "__main__":
    exit(checkmatrix_app())  # pragma: no cover
