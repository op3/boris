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

"""Utils for input and output preparation."""

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def numpy_to_root_hist(
    hist: np.ndarray,
    bin_edges: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]],
):
    """
    Convert numpy histogram with bin edges to root histogram (TH1 or TH2)
    Args:
        hist: histogram
        bin_edges: numpy array or tuple of numpy arrays containing bin edges
    Returns:
        root histogram with bin_edges
    """
    # TODO: Discard sumw2?
    if hist.ndim == 1:
        from uproot3_methods.classes.TH1 import from_numpy

        h = from_numpy([hist, bin_edges])
    else:
        from uproot3_methods.classes.TH2 import from_numpy

        if isinstance(bin_edges, (list, tuple)):
            h = from_numpy([hist, *bin_edges])
        else:
            h = from_numpy([hist, np.arange(0, hist.shape[0] + 1), bin_edges])

    # TODO: Is there a more elegant way to do this in uproot4?
    for attr in [
        "_fTSumwx",
        "_fTsumwx2",
        "_fTsumwy",
        "_fTsumwy2",
        "_fTsumwxy",
        "_fTsumw",
        "_fTsumw2",
        "_fSumw2",
    ]:
        try:
            delattr(h, attr)
        except AttributeError:
            pass
    return h


def _bin_edges_dict(
    bin_edges: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Create dictionary of multiple bin_edges arrays for file formats
    that don’t support expliciting writing bin_edges.
    """
    if isinstance(bin_edges, (list, tuple)):
        return {
            f"bin_edges_{i}": bin_edges_i
            for i, bin_edges_i in enumerate(bin_edges)
        }
    else:
        return {"bin_edges": bin_edges}


def write_hist(
    histfile: Path,
    name: str,
    hist: np.ndarray,
    bin_edges: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
) -> None:
    """Write single histogram to file"""
    if histfile.exists():
        raise Exception(f"Error writing {histfile}, already exists.")
    if histfile.suffix == ".npz":
        np.savez_compressed(
            histfile,
            **{
                name: hist,
                **_bin_edges_dict(bin_edges),
            },
        )
    elif histfile.suffix == ".txt":
        if hist.ndim > 1:
            raise Exception(
                f"Writing {hist.ndim}-dimensional histograms to txt not supported."
            )
        # TODO: Store as (bin_edges_lower, bin_edges_upper, hist) instead
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
                for key, arr in _bin_edges_dict(bin_edges).items():
                    f.create_dataset(
                        key,
                        data=arr,
                        compression="gzip",
                        compression_opts=9,
                    )
        except ModuleNotFoundError as err:
            logger.error("Please install h5py to write hdf5 files")
            raise err
    elif histfile.suffix == ".root":
        import uproot3 as uproot

        with uproot.recreate(
            histfile, compression=uproot.write.compress.LZMA(6)
        ) as f:
            f[name] = numpy_to_root_hist(hist, bin_edges)
    else:
        raise Exception(f"Unknown output format: {histfile.suffix}")


def write_hists(
    hists: Dict[str, np.ndarray],
    bin_edges: Union[np.ndarray, List[np.ndarray]],
    output_path: Path,
) -> None:
    """Write multiple histograms to file."""
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
        hists.update(_bin_edges_dict(bin_edges))
        np.savez_compressed(output_path, **hists)
    elif output_path.suffix == ".root":
        import uproot3 as uproot

        with uproot.recreate(
            output_path, compression=uproot.write.compress.LZMA(6)
        ) as f:
            print(f.compression)
            print(f.compression)
            for key, outspec in hists.items():
                f[key] = numpy_to_root_hist(outspec, bin_edges)
    elif output_path.suffix == ".hdf5":
        try:
            import h5py

            with h5py.File(output_path, "w") as f:
                hists.update(_bin_edges_dict(bin_edges))
                for key, outspec in hists.items():
                    f.create_dataset(
                        key,
                        data=outspec,
                        compression="gzip",
                        compression_opts=9,
                    )
        except ModuleNotFoundError as err:
            logger.error("Please install h5py to write hdf5 files")
            raise err
    else:
        raise Exception(f"File format {output_path.suffix} not supported.")


def get_filetype(path: Path) -> Optional[str]:
    """Determine file format of path using magic bytes"""
    with open(path, "rb") as f:
        header = f.read(4)
        return {
            b"PK\x03\x04": "application/zip",
            b"root": "application/root",
            b"\x89HDF": "application/x-hdf5",
        }.get(header, None)


def get_bin_edges(
    hist: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        if (hist[:-1, 1] == hist[1:, 0]).all():
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


def get_obj_by_name(
    mapping: Mapping[str, Any], name: Optional[str] = None
) -> Any:
    """Return object from mapping

    If a name is provided, the object is returned by name. If no name
    is provided and the mapping contains only one object, this object
    is returned.

    Args:
        mapping: Mapping such as dictionary or uproot-file
        name: Optional name of object to return
    Returns:
        Object, if a unique mapping is found
    """
    if name:
        return mapping[name]
    keys = [key for key in mapping.keys() if not key.startswith("bin_edges")]

    if len(keys) == 1:
        return mapping[keys[0]]
    raise KeyError("Please provide name of object")


def get_obj_bin_edges(mapping: Mapping[str, Any]) -> List[np.ndarray]:
    if "bin_edges" in mapping:
        return [mapping["bin_edges"]]
    bin_edges_keys = sorted(
        [key for key in mapping.keys() if key.startswith("bin_edges_")]
    )
    if bin_edges_keys:
        return [mapping[key] for key in bin_edges_keys]
    raise KeyError("Object does not contain bin_edges.")


def get_keys_in_container(path: Path) -> List[str]:
    """Return all keys that are available in container"""
    filetype = get_filetype(path)
    if filetype == "application/root":
        import uproot

        with uproot.open(path) as container:
            return container.iterkeys(cycle=False)
    elif filetype == "application/zip":
        with np.load(path) as container:
            return container.keys()
    elif filetype == "application/x-hdf5":
        try:
            import h5py

            with h5py.File(path, "r") as container:
                return container.keys()
        except ModuleNotFoundError as err:
            logger.error("Please install h5py to read .hdf5 files")
            raise err
    return []


def read_spectrum(
    spectrum: Path,
    histname: Optional[str] = None,
    bin_edges: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Read spectrum to numpy array.

    Args:
        spectrum: Path of spectrum file
        histname: Name of histogram in spectrum file to load (optional)
        bin_edges: Determine bin edges of spectrum

    Returns:
        Extracted histogram
        Bin edges of spectrum or None, if not available
    """
    filetype = get_filetype(spectrum)
    if filetype == "application/root":
        import uproot

        with uproot.open(spectrum) as specfile:
            return get_obj_by_name(specfile, histname).to_numpy()
    elif filetype == "application/zip":
        with np.load(spectrum) as specfile:
            hist = get_obj_by_name(specfile, histname)
            if bin_edges:
                try:
                    return (hist, get_obj_bin_edges(specfile))
                except KeyError:
                    pass
    elif filetype == "application/x-hdf5":
        try:
            import h5py

            with h5py.File(spectrum, "r") as specfile:
                hist = get_obj_by_name(specfile, histname)[()]
                if bin_edges:
                    try:
                        return (
                            hist,
                            [a[()] for a in get_obj_bin_edges(specfile)],
                        )
                    except KeyError:
                        pass
        except ModuleNotFoundError as err:
            logger.error("Please install h5py to read .hdf5 files")
            raise err
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
    filter_spectrum: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read and rebin spectrum.

    Args:
        spectrum: Path to file containing spectrum
        bin_edges_rebin: Rebin spectrum to this bin_edges array
        histname: Provide object name for files containing multiple objects
        cal_bin_centers: Bin centers to calibrate spectrum before rebinning
        cal_bin_edges: Bin edges to calibrate spectrum before rebinning
        filter_spectrum: Apply filter function to spetrum before rebinning
    Returns:
        Histogram
        Bin edges
    """
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

    if filter_spectrum is not None:
        spectrum = filter_spectrum(spectrum)

    if spectrum_bin_edges is not None:
        spectrum = rebin_uniform(spectrum, spectrum_bin_edges, bin_edges_rebin)

    return spectrum, spectrum_bin_edges


def hdi(
    sample: np.ndarray,
    hdi_prob: Union[float, np.number] = np.math.erf(np.sqrt(0.5)),
) -> Tuple[float, float]:
    """
    Calculate highest density interval (HDI) of sample for given probability
    """
    sample.sort(axis=0)
    interval = int(np.ceil(len(sample) * (1 - hdi_prob)))
    candidates = sample[-interval:] - sample[:interval]
    best = candidates.argmin()
    return (sample[best], sample[best - interval])


def rebin_hist(
    hist: np.ndarray,
    bin_width: int,
    left: Optional[int] = 0,
    right: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin hist with dimension $N^M$.

    The binning is reduced by a factor of bin_width, i.e. neighboring
    bins are summed. Bin edges are assumed to be at [0, 1, 2, …].

    Args:
        hist: Input matrix of type $N^M$ (N bins, M dimensions)
        bin_width: rebinning factor
        left: lower edge of first bin of resulting matrix
        right: maximum upper edge of last bin of resulting matrix

    Returns:
        rebinned matrix
        resulting bin edges
    """
    left = left or 0
    right = right or hist.shape[0] + 1
    num_dim = hist.ndim
    num_bins = (right - left) // bin_width
    upper = left + num_bins * bin_width

    if not (np.array(hist.shape)[1:] == np.array(hist.shape)[:-1]).all():
        raise ValueError("Input histogram has to be square.")

    res = (
        hist[(slice(left, upper),) * num_dim]
        .reshape(*[num_bins, bin_width] * num_dim)
        .sum(axis=(num_dim * 2 - 1))
    )
    for i in range(1, num_dim):
        res = res.sum(axis=i)
    bin_edges = np.linspace(left, upper, num_bins + 1)
    return res, bin_edges


def rebin_uniform(
    hist: np.ndarray, bin_edges: np.ndarray, bin_edges_new: np.ndarray
) -> np.ndarray:
    """Rebin hist from binning bin_edges to bin_edges_new.

    Each count in the original histogram is randomly placed within
    the limits of the corresponding bin following a uniform probability
    distribution. A new histogram with bin edges bin_edges_new is
    filled with the resulting data.

    Args:
        hist: Original histogram to rebin.
        bin_edges: bin edges of original histogram.
        bin_edges_new: bin edges of rebinned histogram.

    Returns:
        rebinned histogram
    """
    if not issubclass(hist.dtype.type, np.integer):
        raise ValueError("Histogram has to be of type integer")
    if not (hist >= 0).all():
        raise ValueError("Histogram contains negative bins")

    data_new = np.random.uniform(
        np.repeat(bin_edges[:-1], hist), np.repeat(bin_edges[1:], hist)
    )
    return np.histogram(data_new, bin_edges_new)[0]


def load_rema(
    path: Path,
    hist_rema: str,
    hist_norm: Optional[str],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load detector response matrix from file.

    Args:
        path: Path of container file containing response matrix
        hist_rema : Name of histogram for response matrix
        hist_norm: Name of histogram used for normalization
            (e.g. containing number of simulated particles)

    Returns:
        response matrix
        List of bin_edges arrays
    """
    rema = read_spectrum(path, hist_rema)
    if hist_norm:
        norm = read_spectrum(path, hist_norm)
        # TODO: Is this the correct axis of rema?
        if not (rema[1][0] == norm[1][0]).all():
            raise Exception(
                "Binning of response matrix and normalization histogram not equal"
            )
        rema /= norm[0]
    return rema


def get_rema(
    path: Union[str, Path], bin_width: int, left: int, right: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtain the response matrix from the root file at path.
    root file has to contain "rema" and "n_simulated_particles" (TH1).
    The response matrix is cropped to left and right and rebinned to
    bin_width.

    Args:
        path: path of root file
        bin_width: rebin matrix to this width
        left: lower boundary of cropped matrix
        right: maximum upper boundary of cropped matrix.

    Returns:
        response matrix
        number of simulated particles
        bin_edges
    """
    rema, bin_edges = load_rema(path, hist_rema, hist_norm)
    if not (
        np.isclose(np.diff(bin_edges[0]), 1.0).all()
        and np.isclose(np.diff(bin_edges[0]), 1.0).all()
    ):
        raise NotImplementedError(
            "Response matrix with binning unequal to 1 keV not yet implemented."
        )

    # TODO:
    rema_re = rebin_hist(rema, bin_width, left, right)
    return rema_re
