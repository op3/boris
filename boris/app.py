#!/usr/bin/env python
#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2020 O. Papst.
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

"""boris command line interface"""

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

if __name__ == "__main__":
    project_dir = Path(__file__).absolute().parents[1].resolve()
    project_path = str((project_dir / "boris").resolve())
    if project_path in sys.path:
        sys.path.remove(project_path)
    sys.path.insert(0, str(project_dir))

from boris import rebin_hist, rebin_uniform, get_rema, deconvolute

logger = logging.getLogger("boris")


def setup_logging():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


@contextlib.contextmanager
def do_step(text: str, simple: bool = False) -> Generator[None, None, None]:
    if not simple:
        logger.info(f"{text} ...")
    try:
        yield
        if simple:
            logger.info(f"{text}")
        else:
            logger.info(f"{text} complete")
    except BaseException as e:
        logger.error(f"{text} failed:")
        logger.error(e, exc_info=True)
        exit(1)


def write_hist(
    histfile: Path, name: str, hist: np.array, bin_edges: np.array
) -> None:
    if histfile.exists():
        raise Exception(f"Error writing {histfile}, already exists.")
    if histfile.suffix == ".npz":
        np.savez_compressed(histfile, **{name: hist, "bin_edges": bin_edges,})
    elif histfile.suffix == ".txt":
        np.savetxt(
            histfile,
            hist,
            header="bin edges:\n" + str(bin_edges) + f"\n{name}:",
        )
    elif histfile.suffix == ".hdf5":
        try:
            import h5py

            with h5py.File(histfile, "w") as f:
                f.create_dataset(
                    name, data=hist, compression="gzip", compression_opts=9
                )
                f.create_dataset(
                    "bin_edges",
                    data=bin_edges,
                    compression="gzip",
                    compression_opts=9,
                )
        except ModuleNotFoundError:
            raise Exception("Please install h5py to write hdf5 files")
    elif histfile.suffix == ".root":
        import uproot

        # TODO: Discard sumw2?
        if hist.ndim == 1:
            from uproot_methods.classes.TH1 import from_numpy

            h = from_numpy([hist, bin_edges])
        else:
            from uproot_methods.classes.TH2 import from_numpy

            h = from_numpy([hist, np.arange(0, hist.shape[0] + 1), bin_edges])
        with uproot.create(histfile) as f:
            f[name] = h
    else:
        raise Exception(f"Unknown output format: {histfile.suffix}")


def get_bin_edges(
    hist: np.ndarray,
) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
    """
    Try to determine if hist contains binning information.

    Args:
        hist: histogram from txt file
    
    Returns:
        Extracted histogram
        bin edges of histogram or None, if not available
    """
    if len(hist.shape) == 1:
        return hist, None
    if len(hist.shape) > 2:
        raise ValueError("Array contains too many dimensions (> 2).")
    if hist.shape[0] < hist.shape[1]:
        hist = hist.T
    if hist.shape[1] >= 3:
        if hist[:-1, 1] == hist[1:, 0].all():
            return hist[:, 2], np.concatenate([hist[:, 0], [hist[-1, 1]]])
        else:
            raise ValueError(
                "Lower and upper bin edges (column 0 and 1) are not continuous."
            )
    if hist.shape[1] == 2:
        diff = np.diff(hist[:, 0]) / 2
        return (
            hist[:, 1],
            np.concatenate(
                [
                    [hist[0, 0] - diff[0]],
                    hist[1:, 0] - diff,
                    [hist[-1, 0] + diff[-1]],
                ]
            ),
        )


def read_spectrum(
    spectrum: Path, histname: Union[None, str] = None
) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
    """
    Read spectrum to numpy array

    Args:
        spectrum: Path of spectrum file
        histname: Name of histogram in spectrum file to load (optional)
    
    Returns:
        Extracted histogram
        Bin edges of spectrum or None, if not available
    """
    if spectrum.suffix == ".root":
        import uproot

        with uproot.open(spectrum) as specfile:
            if histname:
                return specfile[histname].numpy()
            elif len(list(specfile.keys())) == 1:
                return specfile[list(specfile.keys())[0]].numpy()
            else:
                raise Exception(
                    "Please provide name of histogram to read from root file."
                )
    elif spectrum.suffix == ".npz":
        with np.load(spectrum) as specfile:
            if histname:
                hist = specfile[histname]
            elif len(list(specfile.keys())) == 1:
                hist = specfile[list(specfile.keys())[0]]
            else:
                raise Exception(
                    "Please provide name of histogram to read from npz file."
                )
    elif spectrum.suffix == ".hdf5":
        import h5py

        with h5py.File(spectrum) as specfile:
            if histname:
                hist = specfile[histname]
            elif len(list(specfile.keys())) == 1:
                hist = specfile[list(specfile.keys())[0]]
            else:
                raise Exception(
                    "Please provide name of histogram to read from hdf5 file."
                )
    else:
        hist = np.loadtxt(spectrum)
    return get_bin_edges(hist)


def read_pos_int_spectrum(
    spectrum: Path, histname: Union[None, str] = None, tol: float = 0.001
) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
    """
    Read spectrum to numpy array of type np.integer. The spectrum must
    contain no negative bins.

    Args:
        spectrum: Path of spectrum file
        histname: Name of histogram in spectrum file to load (optional)
        tol: Maximum sum area difference for integer conversion (optional)
    
    Returns:
        Extracted histogram
        Bin edges of spectrum or None, if not available
    """
    spectrum, spectrum_bin_edges = read_spectrum(spectrum, histname)
    if not issubclass(spectrum.dtype.type, np.integer):
        logger.warning("Histogram is not of type integer, converting ...")
        sum_before = np.sum(spectrum)
        spectrum = spectrum.astype(np.int64)
        sum_diff = np.sum(spectrum) - sum_before
        if sum_diff > tol:
            raise ValueError(
                "Histogram area changed by more than {sum_diff:e}."
            )
    if (spectrum < 0).any():
        raise ValueError(
            "Assuming Poisson distribution, but histogram contains negative bins."
        )
    return spectrum, spectrum_bin_edges


def read_rebin_spectrum(
    spectrum: Path,
    bin_edges_rebin: np.ndarray,
    histname: Optional[str] = None,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    spectrum, spectrum_bin_edges = read_pos_int_spectrum(spectrum, histname)
    if spectrum_bin_edges is not None and (cal_bin_centers or cal_bin_edges):
        logger.warning("Ignoring calibration, binnig already provided")
    if cal_bin_centers is not None:
        spectrum_bin_edges = np.poly1d(np.array(cal_bin_centers)[::-1])(
            np.arange(spectrum.size + 1) - 0.5
        )
    elif cal_bin_edges is not None:
        spectrum_bin_edges = np.poly1d(np.array(cal_bin_edges)[::-1])(
            np.arange(spectrum.size + 1)
        )

    if spectrum_bin_edges is not None:
        spectrum = rebin_uniform(spectrum, spectrum_bin_edges, bin_edges_rebin)

    return spectrum, spectrum_bin_edges


def boris(
    matrix: Path,
    observed_spectrum: Path,
    incident_spectrum: Path,
    bin_width: int,
    left: int,
    right: int,
    ndraws: int,
    tune: int,
    thin: int,
    burn: int,
    cores: int,
    histname: Optional[str] = None,
    background_spectrum: Optional[Path] = None,
    background_name: Optional[str] = None,
    background_scale: float = 1.0,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
) -> None:
    """
    Load response matrix and spectrum, sample MCMC chain,
    write resulting trace to file.

    Args:
        matrix: Path of response matrix in root format
        observed_spectrum: Read observed spetcrum from this path
        incident_spectrum: Write incident spectrum trace to this path
        histname: name of histogram in observed_spectrum to read (optional)
    """

    with do_step(f"Reading response matrix {matrix}"):
        rema_counts, rema_nsim, rema_bin_edges = get_rema(
            matrix, bin_width, left, right
        )
        rema = rema_counts / rema_nsim

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading observed spectrum {observed_spectrum}{print_histname}"
    ):
        spectrum, spectrum_bin_edges = read_rebin_spectrum(
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

    with do_step(f"🎲 Sampling from posterior distribution"):
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
        args = self.parse_args(sys.argv[1:])
        if args.seed:
            with do_step(f"Setting numpy seed to {args.seed}", simple=True):
                np.random.seed(int(args.seed))
        boris(
            args.matrix,
            args.observed_spectrum,
            args.incident_spectrum,
            args.bin_width,
            args.left,
            args.right,
            args.ndraws,
            args.tune,
            args.thin,
            args.burn,
            args.cores,
            args.hist,
            args.bg_spectrum,
            args.bg_hist,
            args.bg_scale,
            args.cal_bin_centers,
            args.cal_bin_edges,
        )

    def parse_args(self, args: List[str]) -> argparse.Namespace:
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
            "--bin-width",
            help="bin width of deconvoluted spectrum",
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
            "matrix",
            help="response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms",
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

        return parser.parse_args(args)


def sirob(
    matrix: Path,
    incident_spectrum: Path,
    observed_spectrum: Path,
    bin_width: int,
    left: int,
    right: int,
    histname: Optional[str] = None,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
) -> None:
    with do_step(f"Reading response matrix from {matrix}"):
        rema_counts, rema_nsim, rema_bin_edges = get_rema(
            matrix, bin_width, left, right
        )
        rema = rema_counts / rema_nsim

    with do_step(f"Reading incident spectrum from {incident_spectrum}"):
        incident, spectrum_bin_edges = read_rebin_spectrum(
            incident_spectrum,
            rema_bin_edges,
            histname,
            cal_bin_centers,
            cal_bin_edges,
        )

    with do_step("Calculating observed (convoluted) spectrum"):
        observed = incident @ rema

    with do_step(f"Writing observed spectrum to {observed_spectrum}"):
        write_hist(observed_spectrum, "observed", observed, spectrum_bin_edges)


class SirobApp:
    def __init__(self) -> None:
        args = self.parse_args(sys.argv[1:])
        sirob(
            args.matrix,
            args.incident_spectrum,
            args.observed_spectrum,
            args.bin_width,
            args.left,
            args.right,
            args.hist,
            args.cal_bin_centers,
            args.cal_bin_edges,
        )

    def parse_args(self, args: List[str]) -> argparse.Namespace:
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
            "--bin-width",
            help="bin width of deconvoluted spectrum",
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
            "matrix",
            help="response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms",
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
        return parser.parse_args(args)


if __name__ == "__main__":
    setup_logging()
    if Path(sys.argv[0]).stem == "sirob":
        SirobApp()
    else:
        BorisApp()
