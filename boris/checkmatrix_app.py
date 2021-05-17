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
import sys
from pathlib import Path
from typing import List


class CheckMatrixApp:
    """CLI interface for boris2spec."""

    def __init__(self) -> None:
        self.parse_args(sys.argv[1:])
        from boris.app import setup_logging, check_matrix

        setup_logging()
        check_matrix(
            self.args.matrixfile,
            self.args.binning_factor,
            self.args.left,
            self.args.right,
            self.args.rema_name,
            self.args.norm_hist,
        )

    def parse_args(self, args: List[str]):
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            description="Display detector response matrix."
        )
        parser.add_argument(
            "matrixfile",
            help="container file containing detector response matrix",
            type=Path,
        )
        parser.add_argument(
            "-l",
            "--left",
            help="lower edge of first bin of deconvoluted spectrum (default: %(default)s)",
            type=float,
            default=0,
        )
        parser.add_argument(
            "-r",
            "--right",
            help="maximum upper edge of last bin of deconvoluted spectrum (default: maximum energy of simulation)",
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
        parser.add_argument(
            "--norm-hist",
            help="divide detector response matrix by this histogram (e. g., to correct for number of simulated particles) (optional) (default: %(default)s)",
            nargs="?",
            default=None,
            type=str,
        )
        self.args = parser.parse_args(args)


def init():
    """Run app if executed directly."""
    if __name__ == "__main__":
        CheckMatrixApp()


init()
