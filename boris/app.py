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

"""boris command line interface."""

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

if __name__ == "__main__":
    project_dir = Path(__file__).absolute().parents[1].resolve()
    project_path = str((project_dir / "boris").resolve())
    if project_path in sys.path:
        sys.path.remove(project_path)
    sys.path.insert(0, str(project_dir))

from boris.core import deconvolute
from boris.utils import (
    get_rema,
    write_hist,
    write_hists,
    read_rebin_spectrum,
    read_spectrum,
    hdi,
)

logger = logging.getLogger(__name__)


def setup_logging():
    """Prepare logger, set message format"""
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


@contextlib.contextmanager
def do_step(text: str, simple: bool = False) -> Generator[None, None, None]:
    """Contextmanager to print helpful progress messages"""
    if not simple:
        logger.info(f"{text} ...")
    try:
        yield
        if simple:
            logger.info(text)
        else:
            logger.info(f"{text} complete")
    except BaseException as e:
        logger.error(f"{text} failed:")
        logger.error(e, exc_info=True)
        exit(1)


def boris(
    matrix: Path,
    observed_spectrum: Path,
    incident_spectrum: Path,
    binning_factor: int,
    left: int,
    right: int,
    ndraws: int,
    tune: int,
    thin: int,
    burn: int,
    cores: int,
    histname: Optional[str] = None,
    rema_name: str = "rema",
    background_spectrum: Optional[Path] = None,
    background_name: Optional[str] = None,
    background_scale: float = 1.0,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
    norm_hist: Optional[str] = None,
) -> None:
    """Load response matrix and spectrum, sample MCMC chain,
    write resulting trace to file.

    Args:
        matrix: Path of response matrix in root format
        observed_spectrum: Read observed spetcrum from this path
        incident_spectrum: Write incident spectrum trace to this path
        histname: name of histogram in observed_spectrum to read (optional)
    """

    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema, rema_bin_edges = get_rema(
            matrix, rema_name, binning_factor, left, right, norm_hist
        )

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading observed spectrum {observed_spectrum}{print_histname}"
    ):
        spectrum, _ = read_rebin_spectrum(
            observed_spectrum,
            rema_bin_edges,
            histname,
            cal_bin_centers,
            cal_bin_edges,
        )

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            if background_spectrum is not None:
                background, _ = read_rebin_spectrum(
                    background_spectrum,
                    rema_bin_edges,
                    background_name,
                    cal_bin_centers,
                    cal_bin_edges,
                )
            else:
                background, _ = read_rebin_spectrum(
                    observed_spectrum,
                    rema_bin_edges,
                    background_name,
                    cal_bin_centers,
                    cal_bin_edges,
                )

    with do_step("🎲 Sampling from posterior distribution"):
        trace = deconvolute(
            rema,
            spectrum,
            background,
            background_scale,
            ndraws=ndraws,
            tune=tune,
            thin=thin,
            burn=burn,
            cores=cores,
        )

    with do_step(f"💾 Writing incident spectrum trace to {incident_spectrum}"):
        write_hist(
            incident_spectrum, "incident", trace["incident"], rema_bin_edges
        )


class BorisApp:
    def __init__(self) -> None:
        """CLI interface for boris."""
        self.parse_args(sys.argv[1:])
        if self.args.seed:
            with do_step(
                f"Setting numpy seed to {self.args.seed}", simple=True
            ):
                np.random.seed(int(self.args.seed))
        boris(
            self.args.matrix,
            self.args.observed_spectrum,
            self.args.incident_spectrum,
            self.args.binning_factor,
            self.args.left,
            self.args.right,
            self.args.ndraws,
            self.args.tune,
            self.args.thin,
            self.args.burn,
            self.args.cores,
            self.args.hist,
            self.args.bg_spectrum,
            self.args.bg_hist,
            self.args.bg_scale,
            self.args.cal_bin_centers,
            self.args.cal_bin_edges,
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


def sirob(
    matrix: Path,
    incident_spectrum: Path,
    observed_spectrum: Path,
    binning_factor: int,
    left: int,
    right: int,
    histname: Optional[str] = None,
    rema_name: str = "rema",
    background_spectrum: Optional[Path] = None,
    background_name: Optional[str] = None,
    background_scale: float = 1.0,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
    norm_hist: Optional[str] = None,
) -> None:
    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema, rema_bin_edges = get_rema(
            matrix, rema_name, binning_factor, left, right, norm_hist
        )

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading incident spectrum {incident_spectrum}{print_histname}"
    ):
        incident, spectrum_bin_edges = read_rebin_spectrum(
            incident_spectrum,
            rema_bin_edges,
            histname,
            cal_bin_centers,
            cal_bin_edges,
        )

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            if background_spectrum is not None:
                background, _ = read_rebin_spectrum(
                    background_spectrum,
                    rema_bin_edges,
                    background_name,
                    cal_bin_centers,
                    cal_bin_edges,
                )
            else:
                background, _ = read_rebin_spectrum(
                    observed_spectrum,
                    rema_bin_edges,
                    background_name,
                    cal_bin_centers,
                    cal_bin_edges,
                )

    with do_step("Calculating observed (convoluted) spectrum"):
        observed = incident @ rema
        if background:
            observed += background_scale * background

    with do_step(f"Writing observed spectrum to {observed_spectrum}"):
        write_hist(observed_spectrum, "observed", observed, spectrum_bin_edges)


class SirobApp:
    def __init__(self) -> None:
        """CLI interface for sirob."""
        self.parse_args(sys.argv[1:])
        sirob(
            self.args.matrixfile,
            self.args.rema_name,
            self.args.incident_spectrum,
            self.args.observed_spectrum,
            self.args.binning_factor,
            self.args.left,
            self.args.right,
            self.args.hist,
            self.args.bg_spectrum,
            self.args.bg_hist,
            self.args.bg_scale,
            self.args.cal_bin_centers,
            self.args.cal_bin_edges,
            self.args.norm_hist,
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
            "--binning-facor",
            help="rebinning factor, group this many bins together",
            type=int,
            default=10,
        )
        parser.add_argument(
            "-H",
            "--hist",
            help="Name of histogram in incident_spectrum to read (optional)",
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
            help="Provide an energy calibration for the bin centers of the incident spectrum, if bins are unknown (tv style calibration)",
            type=float,
            nargs="+",
        )
        calgroup.add_argument(
            "--cal-bin-edges",
            metavar=("C0", "C1"),
            help="Provide an energy calibration for the bin edges of the incident spectrum, if bins are unknown",
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
            "incident_spectrum",
            help="file containing the incident spectrum",
            type=Path,
        )
        parser.add_argument(
            "observed_spectrum",
            help="write observed (convoluted) spectrum to this path",
            type=Path,
        )
        self.args = parser.parse_args(args)


def boris2spec(
    incident_spectrum: Path,
    output_path: Optional[Path] = None,
    plot: bool = False,
    get_mean: bool = False,
    get_median: bool = False,
    # get_mode: bool = False,
    get_variance: bool = False,
    get_std_dev: bool = False,
    get_hdi: bool = False,
    hdi_prob: float = np.math.erf(np.sqrt(0.5)),
) -> None:
    """Create spectra from boris trace file

    Args:
        incident_spectrum: boris output for incident spectrum
        output_path: Optionally write generated spectra to file
        plot: Optionally display matplotlib window of all spectra
        get_mean: Generate spectrum containing mean of each bin
        get_median: Generate spectrum containing median of each bin
        get_variance: Generate spectrum containing variane of each bin
        get_std_dev: Generate spectrum containing standard deviation of each bin
        get_hdi: Generate spectra containing highest density interval of each bin
        hdi_prob: Probability for which the highest density interval will be computed
    """
    spec, bin_edges = read_spectrum(incident_spectrum, "incident")
    bin_edges = bin_edges[-1]

    res = {}

    if get_mean:
        res["mean"] = np.mean(spec, axis=0)

    if get_median:
        res["median"] = np.median(spec, axis=0)

    # if get_mode:
    #    raise NotImplementedError("mode not yet implemented")
    #    #res["mode"] = get_mode(spec, axis=0)

    if get_variance:
        res["var"] = np.var(spec, axis=0)

    if get_std_dev:
        res["std"] = np.std(spec, axis=0)

    if get_hdi:
        res["hdi_lo"], res["hdi_hi"] = hdi(spec, hdi_prob=hdi_prob)

    if plot:
        import matplotlib.pyplot as plt

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if get_hdi:
            plt.fill_between(
                bin_centers,
                res["hdi_lo"],
                res["hdi_hi"],
                label="Highest Density Interval",
                step="mid",
                alpha=0.3,
            )

        label = {
            "mean": "Mean",
            "median": "Median",
            "mode": "Mode",
            "std": "Standard Deviation",
            "var": "Variance",
        }

        for key in label.keys():
            if key in res:
                plt.step(bin_centers, res[key], where="mid", label=label[key])

        plt.legend()
        plt.tight_layout()
        plt.show()

    if output_path:
        write_hists(res, bin_edges, output_path)


class Boris2SpecApp:
    def __init__(self) -> None:
        """CLI interface for boris2spec."""
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
            help="get the mean for each bin",
            action="store_true",
        )
        # parser.add_argument(
        #    "--get-mode",
        #    help="get the mode for each bin. Requires a lot of statistics to be sufficiently robust",
        #    action="store_true",
        # )
        parser.add_argument(
            "--get-median",
            help="get the median for each bin",
            action="store_true",
        )
        parser.add_argument(
            "--get-variance",
            help="get the variance for each bin",
            action="store_true",
        )
        parser.add_argument(
            "--get-std-dev",
            help="get the standard deviation for each bin",
            action="store_true",
        )
        parser.add_argument(
            "--get-hdi",
            help="get the highest density interval for each bin",
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


if __name__ == "__main__":
    setup_logging()
    if Path(sys.argv[0]).stem == "sirob":
        SirobApp()
    elif Path(sys.argv[0]).stem == "boris2spec":
        Boris2SpecApp()
    if Path(sys.argv[0]).stem == "boris":
        BorisApp()
