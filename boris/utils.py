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

"""Utils for input and output preparation."""

import logging
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np

from boris.core import rebin_uniform

logger = logging.getLogger(__name__)


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


def write_hists(
    hists: Dict[str, np.ndarray], bin_edges: np.ndarray, output_path: Path,
) -> None:
    if output_path.suffix == ".txt":
        stem = output_path.stem
        bins_lo = bin_edges[:-1]
        bins_hi = bin_edges[1:]
        for key, outspec in hists.items():
            np.savetxt(
                output_path.parents[0] / f"{stem}_{key}.txt",
                np.array([bins_lo, bins_hi, outspec]).T,
                header=f"bin_edges_low, bin_edges_high, {key}",
            )
    elif output_path.suffix == ".npz":
        hists["bin_edges"] = bin_edges
        np.savez_compressed(output_path, **hists)
    elif output_path.suffix == ".root":
        import uproot
        from uproot_methods.classes.TH1 import from_numpy

        with uproot.create(output_path) as f:
            for key, outspec in hists.items():
                f[key] = from_numpy([outspec, bin_edges])
    elif output_path.suffix == ".hdf5":
        try:
            import h5py

            with h5py.File(output_path, "w") as f:
                for key, outspec in hists.items():
                    f.create_dataset(
                        key,
                        data=outspec,
                        compression="gzip",
                        compression_opts=9,
                    )

                f.create_dataset(
                    "bin_edges",
                    data=bin_edges,
                    compression="gzip",
                    compression_opts=9,
                )
        except ModuleNotFoundError:
            raise Exception("Please install h5py to write hdf5 files")
    else:
        raise Exception(f"File format {output_path.suffix} not supported.")


def get_bin_edges(hist: np.ndarray,) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Try to determine if hist contains binning information.

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
    spectrum: Path, histname: Optional[str] = None, bin_edges: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read spectrum to numpy array

    Args:
        spectrum: Path of spectrum file
        histname: Name of histogram in spectrum file to load (optional)
        bin_edges: Determine bin edges of spectrum

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

        with h5py.File(spectrum, "r") as specfile:
            if histname:
                hist = specfile[histname][()]
            elif len(list(specfile.keys())) == 1:
                hist = specfile[list(specfile.keys())[0]][()]
            else:
                raise Exception(
                    "Please provide name of histogram to read from hdf5 file."
                )
    else:
        hist = np.loadtxt(spectrum)
    if bin_edges:
        return get_bin_edges(hist)
    return hist


def read_pos_int_spectrum(
    spectrum: Path, histname: Optional[str] = None, tol: float = 0.001
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def shortest_coverage_interval(sample, span=np.math.erf(np.sqrt(0.5))):
    sample.sort(axis=0)
    interval = int(np.ceil(len(sample) * (1 - span)))
    candidates = sample[-interval:] - sample[:interval]
    best = candidates.argmin()
    return (sample[best], sample[best - interval])
