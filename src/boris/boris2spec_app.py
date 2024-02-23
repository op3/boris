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

"""boris2spec cli for spectra creation from boris trace files."""

from __future__ import annotations

import argparse
from pathlib import Path

import math


def boris2spec_app():
    """CLI interface for boris2spec."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create spectra from boris trace files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity",
        action="store_true",
    )
    parser.add_argument(
        "--var-names",
        help="Names of variables that are evaluated",
        type=str,
        nargs="*",
        default=["spectrum", "incident_scaled_to_fep"],
    )
    parser.add_argument(
        "--plot",
        help="Generate a matplotlib plot of the queried spectra. The plot is displayed interactively unless an output filename is given.",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="OUTPUT",
    )
    parser.add_argument(
        "--plot-title",
        help="Set plot title",
        type=str,
    )
    parser.add_argument(
        "--plot-xlabel",
        help="Set plot x-axis label",
        type=str,
    )
    parser.add_argument(
        "--plot-ylabel",
        help="Set plot y-axis label",
        type=str,
    )
    parser.add_argument(
        "--get-mean",
        help="Get the mean for each bin",
        action="store_true",
    )
    # parser.add_argument(
    #    "--get-mode",
    #    help="Get the mode for each bin. Requires a lot of statistics to be sufficiently robust",
    #    action="store_true",
    # )
    parser.add_argument(
        "--get-median",
        help="Get the median for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--get-variance",
        help="Get the variance for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--get-std-dev",
        help="Get the standard deviation for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--get-min",
        help="Get the minimum for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--get-max",
        help="Get the maximum for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--get-hdi",
        help="Get the highest density interval for each bin",
        action="store_true",
    )
    parser.add_argument(
        "--hdi-prob",
        metavar="PROB",
        help="HDI prob for which interval will be computed",
        default=math.erf(math.sqrt(0.5)),
    )
    parser.add_argument(
        "--burn",
        metavar="NUMBER",
        help="Skip initial NUMBER of samples to account for burn-in phase during sampling",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--thin",
        metavar="FACTOR",
        help="Thin trace by FACTOR before evaluating to reduce autocorrelation",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--force-overwrite",
        help="Overwrite existing files without warning",
        action="store_true",
    )

    parser.add_argument(
        "trace_file",
        help="boris output containing traces",
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Write resulting spectra to this file (multiple files are created for each exported spectrum if txt format is used)",
        type=Path,
        nargs="?",
    )

    args = parser.parse_args()
    if args.plot is None and args.output_path is None:
        parser.error("Please specify output_path and/or use --plot option")

    if not (
        args.get_mean
        or args.get_median
        # or args.get_mode
        or args.get_variance
        or args.get_std_dev
        or args.get_hdi
    ):
        parser.error("Nothing to do, please give some --get-* options")

    from boris.app import setup_logging, boris2spec

    setup_logging(args.verbose)
    boris2spec(
        args.trace_file,
        args.output_path,
        args.var_names,
        args.plot,
        args.plot_title,
        args.plot_xlabel,
        args.plot_ylabel,
        args.get_mean,
        args.get_median,
        # args.get_mode,
        args.get_variance,
        args.get_std_dev,
        args.get_min,
        args.get_max,
        args.get_hdi,
        args.hdi_prob,
        args.burn,
        args.thin,
        args.force_overwrite,
    )


if __name__ == "__main__":
    exit(boris2spec_app())  # pragma: no cover
