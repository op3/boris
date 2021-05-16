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

"""boris cli for detector response deconvolution"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

from boris.app import do_step, boris


class BorisApp:
    def __init__(self) -> None:
        """CLI interface for boris"""
        self.parse_args(sys.argv[1:])
        if self.args.seed:
            with do_step(
                f"Setting numpy seed to {self.args.seed}", simple=True
            ):
                np.random.seed(int(self.args.seed))
        boris(
            self.args.matrixfile,
            self.args.observed_spectrum,
            self.args.incident_spectrum,
            self.args.binning_factor,
            self.args.left,
            self.args.right,
            self.args.hist,
            self.args.rema_name,
            self.args.bg_spectrum,
            self.args.bg_hist,
            self.args.bg_scale,
            self.args.cal_bin_centers,
            self.args.cal_bin_edges,
            self.args.norm_hist,
            ndraws=self.args.ndraws,
            tune=self.args.tune,
            thin=self.args.thin,
            burn=self.args.burn,
            cores=self.args.cores,
        )

    def parse_args(self, args: List[str]):
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "-l",
            "--left",
            help="lower edge of first bin of deconvoluted spectrum",
            type=int,
            default=0,
        )
        parser.add_argument(
            "-r",
            "--right",
            help="maximum upper edge of last bin of deconvoluted spectrum",
            type=int,
            default=None,
        )
        parser.add_argument(
            "-b",
            "--binning-factor",
            help="rebinning factor, group this many bins together",
            type=int,
            default=10,
        )
        parser.add_argument("-s", "--seed", help="set random seed")
        parser.add_argument(
            "-c",
            "--cores",
            help="number of cores to utilize",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--thin",
            help="thin the resulting trace by a factor",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--tune",
            help="number of initial steps used to tune the model",
            default=1000,
            type=int,
        )
        parser.add_argument(
            "--burn",
            help="number of initial steps to discard (burn-in phase)",
            default=1000,
            type=int,
        )
        parser.add_argument(
            "-n",
            "--ndraws",
            help="number of samples to draw per core",
            default=2000,
            type=int,
        )
        parser.add_argument(
            "-H",
            "--hist",
            help="name of histogram in observed_spectrum to read (optional)",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--bg-spectrum",
            help="path to observed background spectrum (optional)",
            default=None,
            type=Path,
        )
        parser.add_argument(
            "--bg-hist",
            help="name of background histogram in observed_spectrum or --bg-spectrum, if specified (optional)",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--bg-scale",
            help="relative scale of background spectrum live time to observed spectrum live time (optional)",
            default=1.0,
            type=float,
        )

        calgroup = parser.add_mutually_exclusive_group()
        calgroup.add_argument(
            "--cal-bin-centers",
            metavar=("C0", "C1"),
            help="Provide an energy calibration for the bin centers of the observed spectrum, if bins are unknown (tv style calibration)",
            type=float,
            nargs="+",
        )
        calgroup.add_argument(
            "--cal-bin-edges",
            metavar=("C0", "C1"),
            help="Provide an energy calibration for the bin edges of the observed spectrum, if bins are unknown",
            type=float,
            nargs="+",
        )
        parser.add_argument(
            "--rema-name",
            help="Name of the detector response matrix in matrix file",
            default="rema",
            nargs=1,
            type=str,
        )
        parser.add_argument(
            "--norm-hist",
            help="Divide detector response matrix by this histogram (e. g., to correct for number of simulated particles)",
            nargs="?",
            default=None,
            type=str,
        )

        parser.add_argument(
            "matrixfile",
            help="container file containing detector response matrix",
            type=Path,
        )
        parser.add_argument(
            "observed_spectrum",
            help="txt file containing the observed spectrum",
            type=Path,
        )
        parser.add_argument(
            "incident_spectrum",
            help="write trace of incident spectrum to this path",
            type=Path,
        )

        self.args = parser.parse_args(args)


def init():
    if __name__ == "__main__":
        BorisApp()


init()