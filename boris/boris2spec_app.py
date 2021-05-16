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
import sys
from pathlib import Path
from typing import List

import numpy as np

from boris.app import setup_logging, boris2spec


class Boris2SpecApp:
    """CLI interface for boris2spec."""

    def __init__(self) -> None:
        setup_logging()
        self.parse_args(sys.argv[1:])
        boris2spec(
            self.args.incident_spectrum,
            self.args.output_path,
            self.args.plot,
            self.args.get_mean,
            self.args.get_median,
            # self.args.get_mode,
            self.args.get_variance,
            self.args.get_std_dev,
            self.args.get_min,
            self.args.get_max,
            self.args.get_hdi,
            self.args.hdi_prob,
        )

    def parse_args(self, args: List[str]):
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--plot",
            help="Display a matplotlib plot of the queried spectra",
            action="store_true",
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
            default=np.math.erf(np.sqrt(0.5)),
        )

        parser.add_argument(
            "incident_spectrum",
            help="boris output for incident spectrum",
            type=Path,
        )
        parser.add_argument(
            "output_path",
            help="Write resulting spectra to this file (multiple files are created for each exported spectrum if txt format is used)",
            type=Path,
            nargs="?",
        )
        self.args = parser.parse_args(args)
        if not self.args.plot and self.args.output_path is None:
            parser.error("Please specify output_path and/or use --plot option")

        if not (
            self.args.get_mean
            or self.args.get_median
            # or self.args.get_mode
            or self.args.get_variance
            or self.args.get_std_dev
            or self.args.get_hdi
        ):
            parser.error("Nothing to do, please give some --get-* options")


def init():
    """Run app if executed directly"""
    if __name__ == "__main__":
        Boris2SpecApp()


init()
