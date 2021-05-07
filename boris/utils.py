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

from boris.core import rebin_uniform

logger = logging.getLogger(__name__)


def numpy_to_root_hist(
    hist: np.ndarray,
    bin_edges: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]],
) -> "uproot.rootio.ROOTDirectory":
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
    del h._fTsumwx
    del h._fTsumwx2
    del h._fTsumwy
    del h._fTsumwy2
    del h._fTsumwxy
    del h._fTsumw
    del h._fTsumw2
    del h._fSumw2
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
    hist: np.array,
    bin_edges: Union[np.array, List[np.ndarray], Tuple[np.array]],
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
                for key, arr in _bin_edges_dict(bin_edges):
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
            b"\xd3HDF": "application/x-hdf5",
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
                    return [hist, get_obj_bin_edges(specfile)]
                except KeyError:
                    pass
    elif filetype == "application/x-hdf5":
        try:
            import h5py

            with h5py.File(spectrum, "r") as specfile:
                hist = get_obj_by_name(specfile, histname)[()]
                if bin_edges:
                    try:
                        return [
                            hist,
                            [a[()] for a in get_obj_bin_edges(specfile)],
                        ]
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
