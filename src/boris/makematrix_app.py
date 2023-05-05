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

"""makematrix cli for detector response matrix creation."""

from __future__ import annotations

import argparse
from pathlib import Path


def makematrix_app():
    """CLI interface for make_matrix."""
    parser = argparse.ArgumentParser(
        description="Create a detector response matrix from multiple simulated spectra for different energies."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity",
        action="store_true",
    )
    parser.add_argument(
        "--sim-dir",
        help="simulation file names are given relative to this directory (default: Directory containing datfile)",
        type=Path,
    )
    parser.add_argument(
        "--detector",
        nargs="*",
        help="Names of histograms to create response matrices for (default: All available histograms)",
    )
    parser.add_argument(
        "--max-energy",
        nargs="?",
        help="Maximum energy of created response matrix",
        type=float,
    )
    parser.add_argument(
        "--force-overwrite",
        help="Overwrite existing files without warning",
        action="store_true",
    )

    parser.add_argument(
        "datfile",
        help=(
            "datfile containing simulation information, each line has format "
            "`<simulation_hists.root>: <energy> [<number of particles>]`"
        ),
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Write resulting response matrix to this file.",
        type=Path,
    )
    args = parser.parse_args()

    from boris.app import setup_logging, make_matrix

    setup_logging(args.verbose)
    make_matrix(
        args.datfile,
        args.output_path,
        args.detector,
        args.max_energy,
        args.sim_dir,
        args.force_overwrite,
    )


if __name__ == "__main__":
    exit(makematrix_app())  # pragma: no cover
