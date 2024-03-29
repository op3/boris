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

"""boris cli for detector response reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path


def boris_app():
    """CLI interface for boris"""
    parser = argparse.ArgumentParser(
        description="Reconstruct incident_spectrum based on observed_spectrum using the supplied detector response matrix.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--left",
        help="crop spectrum to the lowest bin still containing LEFT (default: %(default)s)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-r",
        "--right",
        help="crop spectrum to the highest bin not containing RIGHT (default: maximum energy of simulation)",
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
        "-H",
        "--hist",
        help="name of histogram in observed_spectrum to read (optional) (default: %(default)s)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--bg-spectrum",
        help="path to observed background spectrum (optional) (default: %(default)s)",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--bg-hist",
        help="name of background histogram in observed_spectrum or --bg-spectrum, if specified (optional)  (default: %(default)s)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--bg-scale",
        help="relative scale of background spectrum live time to observed spectrum live time (optional)  (default: %(default)s)",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--rema-name",
        help="name of the detector response matrix in matrix file  (default: %(default)s)",
        default="rema",
        nargs="?",
        type=str,
    )
    parser.add_argument(
        "--matrixfile-alt",
        help="Load an additional detector response matrix from this matrix file (same rema-name as main matrix). Boris will create a linear combination of the main matrixfile and the alternative matrix file. (default: %(default)s)",
        nargs="?",
        default=None,
        type=Path,
    )

    calgroup = parser.add_mutually_exclusive_group()
    calgroup.add_argument(
        "--cal-bin-centers",
        metavar=("C0", "C1"),
        help="energy calibration for the bin centers of the observed spectrum, if bins are unknown (tv style calibration)  (default: %(default)s)",
        type=float,
        nargs="+",
    )
    calgroup.add_argument(
        "--cal-bin-edges",
        metavar=("C0", "C1"),
        help="energy calibration for the bin edges of the observed spectrum, if bins are unknown  (default: %(default)s)",
        type=float,
        nargs="+",
    )

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument("-s", "--seed", help="set random seed")
    advanced.add_argument(
        "-c",
        "--cores",
        help="number of cores to utilize (default: %(default)s)",
        default=1,
        type=int,
    )
    advanced.add_argument(
        "--tune",
        help="number of initial steps used to tune the model (default: %(default)s)",
        default=1000,
        type=int,
    )
    advanced.add_argument(
        "-n",
        "--ndraws",
        help="number of samples to draw per core (default: %(default)s)",
        default=2000,
        type=int,
    )
    advanced.add_argument(
        "--fit-beam",
        help="Perform a fit of a beam profile (default: %(default)s)",
        action="store_true",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Overwrite existing files without warning",
        action="store_true",
    )

    parser.add_argument(
        "matrixfile",
        help="container file containing detector response matrix",
        type=Path,
    )
    parser.add_argument(
        "observed_spectrum",
        help="file containing the observed spectrum",
        type=Path,
    )
    parser.add_argument(
        "incident_spectrum",
        help="write trace of incident spectrum to this path (.h5 file)",
        type=Path,
    )

    args = parser.parse_args()

    import numpy as np
    from boris.app import do_step, setup_logging, boris

    setup_logging(args.verbose)
    if args.seed:
        with do_step(f"Setting numpy seed to {args.seed}", simple=True):
            np.random.seed(int(args.seed))

    calibration = args.cal_bin_edges or args.cal_bin_centers
    convention = "centers" if args.cal_bin_centers else "edges"
    boris(
        args.matrixfile,
        args.observed_spectrum,
        args.incident_spectrum,
        args.binning_factor,
        args.left,
        args.right,
        args.hist,
        args.rema_name,
        args.bg_spectrum,
        args.bg_hist,
        args.bg_scale,
        calibration,
        convention,
        args.matrixfile_alt,
        args.force_overwrite,
        ndraws=args.ndraws,
        tune=args.tune,
        cores=args.cores,
        fit_beam=args.fit_beam,
    )


if __name__ == "__main__":
    exit(boris_app())  # pragma: no cover
