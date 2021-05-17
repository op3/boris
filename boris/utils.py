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

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def numpy_to_root_hist(
    hist: np.ndarray,
    bin_edges: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]],
) -> Any:
    """
    Converts numpy histogram with bin edges to root histogram (TH1 or TH2)

    :param hist: numpy histogram (1D or 2D)
    :param bin_edges:
        numpy array or tuple of numpy arrays containing bin edges
    :return: Root histogram with bin_edges
    """
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
    Creates dictionary of multiple bin_edges arrays for file formats
    that don’t support explicitly writing bin_edges.

    :param bin_edges: array or list/tuple of arrays containing bin_edges.
    :return:
        Dictionary of bin_edges arrays that can be included in a
        container file.
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
    force_overwrite: bool = False,
) -> None:
    """
    Writes single histogram to file.

    :param histfile: Path of created container file.
    :param name: Name of histogram in container file.
    :param hist: Histogram.
    :param bin_edges: Bin edges.
    :param force_overwrite: Overwrite output_path if it exists.
    """
    # TODO: Remove, only use write_hists()
    if not force_overwrite and histfile.exists():
        raise Exception(f"Error writing {histfile}, already exists.")
    if not histfile.parent.exists():
        histfile.parent.mkdir(parents=True)
    if histfile.suffix == ".npz":
        np.savez_compressed(
            histfile,
            **{
                name: hist,
                **_bin_edges_dict(bin_edges),
            },
        )
    elif histfile.suffix == ".txt":
        if hist.ndim != 1:
            raise Exception(
                f"Writing {hist.ndim}-dimensional histograms to txt not supported."
            )
        np.savetxt(
            histfile,
            np.array([bin_edges[:-1], bin_edges[1:], hist]).T,
            header=f"bin_edges_lo, bin_edges_hi, {name}",
        )
    elif histfile.suffix == ".hdf5":
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
    force_overwrite: bool = False,
) -> None:
    """
    Writes multiple histograms to file.

    :param hists: Dictionary of histograms.
    :param bin_edges: Bin edges, assumed to be equal for all histograms.
    :param output_path: Path of created container file.
    :param force_overwrite: Overwrite output_path if it exists.
    """
    if not force_overwrite and output_path.exists():
        raise Exception(f"Error writing {output_path}, already exists.")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    if output_path.suffix == ".txt":
        for hist in hists.values():
            if hist.ndim != 1:
                raise Exception(
                    f"Writing {hist.ndim}-dimensional histograms to txt not supported."
                )
        np.savetxt(
            output_path,
            np.array([bin_edges[:-1], bin_edges[1:], *hists.values()]).T,
            header=", ".join(
                ["bin_edges_lo", "bin_edges_hi"] + list(hists.keys())
            ),
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
    else:
        raise Exception(f"File format {output_path.suffix} not supported.")


def get_filetype(path: Path) -> Optional[str]:
    """
    Determines file format of path using magic bytes.
    Supports root-files, hdf5 and non-empty zip-files (used by npz).

    :param path: Path of file.
    :return: Mimetype of file or ``None``, if not sure.
    """
    with open(path, "rb") as f:
        header = f.read(4)
        return {
            b"PK\x03\x04": "application/zip",
            b"root": "application/root",
            b"\x89HDF": "application/x-hdf5",
        }.get(header, None)


def get_bin_edges(
    hist: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Trys to determine if 2D hist contains column(s) with binning information.

    :param hist: Histogram

    :return:
        - Extracted histogram
        - Bin edges of histogram or None, if not available
    :raises: ValueError
    """
    if hist.ndim != 2 or hist.shape[1] < 2:
        raise ValueError(
            "Array is wrong dimension (≠ 2), cannot extract bin edges."
        )
    if hist.shape[0] < hist.shape[1]:
        hist = hist.T
    if hist.shape[1] >= 3:
        if (hist[:-1, 1] == hist[1:, 0]).all():
            return hist[:, 2], [np.concatenate([hist[:, 0], [hist[-1, 1]]])]
        else:
            raise ValueError(
                "Lower and upper bin edges (column 0 and 1) are not continuous."
            )
    diff = np.diff(hist[:, 0]) / 2
    return (
        hist[:, 1],
        [
            np.concatenate(
                [
                    [hist[0, 0] - diff[0]],
                    hist[1:, 0] - diff,
                    [hist[-1, 0] + diff[-1]],
                ]
            )
        ],
    )


def get_obj_by_name(
    mapping: Mapping[str, Any], name: Optional[str] = None
) -> Any:
    """
    Returns object from mapping.

    If a name is provided, the object is returned by name. If no name
    is provided and the mapping contains only one object, this object
    is returned.

    :param mapping: Mapping such as dictionary or container file.
    :param name: Optional name of object to return.
    :return: Object, if a unique mapping is found.
    """
    if name:
        return mapping[name]
    keys = [key for key in mapping.keys() if not key.startswith("bin_edges")]

    if len(keys) == 1:
        return mapping[keys[0]]
    raise KeyError("Please provide name of object")


def get_obj_bin_edges(mapping: Mapping[str, Any]) -> List[np.ndarray]:
    """
    Returns bin edges from mapping.

    :param mapping: Mapping such as dictionary or container file.
    :return: List of bin edges arrays.
    """
    if "bin_edges" in mapping:
        return [mapping["bin_edges"]]
    bin_edges_keys = sorted(
        [key for key in mapping.keys() if key.startswith("bin_edges_")]
    )
    if bin_edges_keys:
        return [mapping[key] for key in bin_edges_keys]
    raise KeyError("Object does not contain bin_edges.")


def get_keys_in_container(path: Path) -> List[str]:
    """
    Returns all keys that are available in a container.

    :param path: Path of container file.
    :return: List of available keys.
    """
    filetype = get_filetype(path)
    if filetype == "application/root":
        import uproot

        with uproot.open(path) as container:
            return list(container.iterkeys(cycle=False))
    elif filetype == "application/zip":
        with np.load(path) as container:
            return list(container.keys())
    elif filetype == "application/x-hdf5":
        import h5py

        with h5py.File(path, "r") as container:
            return list(container.keys())
    return []


def read_spectrum(
    spectrum: Path,
    histname: Optional[str] = None,
    extract_bin_edges: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reads spectrum to numpy array.

    :param spectrum: Path of spectrum file.
    :param histname: Name of histogram in spectrum file to load (optional).
    :param extract_bin_edges: Determine bin edges of spectrum.

    :return:
        - Extracted histogram.
        - List of bin edges of spectrum, empty if not available.
    """
    filetype = get_filetype(spectrum)
    if filetype == "application/root":
        import uproot

        with uproot.open(spectrum) as specfile:
            hist, *res_bin_edges = get_obj_by_name(
                specfile, histname
            ).to_numpy()
            return hist, res_bin_edges
    elif filetype == "application/zip":
        with np.load(spectrum) as specfile:
            hist = get_obj_by_name(specfile, histname)
            if extract_bin_edges:
                try:
                    return (hist, get_obj_bin_edges(specfile))
                except KeyError:
                    pass
    elif filetype == "application/x-hdf5":
        import h5py

        with h5py.File(spectrum, "r") as specfile:
            hist = get_obj_by_name(specfile, histname)[()]
            if extract_bin_edges:
                try:
                    return (
                        hist,
                        [a[()] for a in get_obj_bin_edges(specfile)],
                    )
                except KeyError:
                    pass
    else:
        hist = np.loadtxt(spectrum)
    if extract_bin_edges:
        return get_bin_edges(hist)
    return hist, []


def read_pos_int_spectrum(
    path: Path,
    histname: Optional[str] = None,
    extract_bin_edges: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reads spectrum to numpy array of type np.integer. The spectrum must
    not contain negative bins.

    :param spectrum: Path of spectrum file.
    :param histname: Name of histogram in spectrum file to load (optional).
    :param extract_bin_edges: Determine bin edges of spectrum.
    :param rtol:
        The relative tolerance parameter (optional, passed to numpy.isclose).
    :param atol:
        The absolute tolerance parameter (optional, passed to numpy.isclose).

    :return:
        - Extracted histogram
        - Bin edges of spectrum or None, if not available
    """
    spectrum, spectrum_bin_edges = read_spectrum(
        path, histname, extract_bin_edges
    )
    if not issubclass(spectrum.dtype.type, np.integer):
        logger.warning("Histogram is not of type integer, converting ...")
        spectrum_before = spectrum
        spectrum = spectrum.astype(np.int64)
        if not np.isclose(
            spectrum_before, spectrum, rtol=rtol, atol=atol
        ).all():
            raise ValueError(
                "Lost precision during conversion to integer histogram."
            )
    if (spectrum < 0).any():
        raise ValueError(
            "Assuming Poisson distribution, but histogram contains negative bins."
        )
    return spectrum, spectrum_bin_edges


def read_rebin_spectrum(
    path: Path,
    bin_edges_rebin: np.ndarray,
    histname: Optional[str] = None,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
    filter_spectrum: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reads and rebin spectrum.

    :param spectrum: Path to file containing spectrum.
    :param bin_edges_rebin: Rebin spectrum to this bin_edges array.
    :param histname:
        Provide object name for files containing multiple objects.
    :param cal_bin_centers:
        Bin centers to calibrate spectrum before rebinning.
    :param cal_bin_edges:
        Bin edges to calibrate spectrum before rebinning.
    :param filter_spectrum:
        Apply filter function to spetrum before rebinning.
    :return:
        - Histogram
        - Bin edges
    """
    if cal_bin_centers or cal_bin_edges:
        spectrum, included_bin_edges = read_pos_int_spectrum(
            path, histname, extract_bin_edges=False
        )
        if included_bin_edges:
            logger.warning("Ignoring binning information of spectrum")
        if cal_bin_centers:
            spectrum_bin_edges = np.poly1d(np.array(cal_bin_centers)[::-1])(
                np.arange(spectrum.size + 1) - 0.5
            )
        else:
            spectrum_bin_edges = np.poly1d(np.array(cal_bin_edges)[::-1])(
                np.arange(spectrum.size + 1)
            )
    else:
        spectrum, (spectrum_bin_edges,) = read_pos_int_spectrum(path, histname)

    if filter_spectrum is not None:
        spectrum = filter_spectrum(spectrum)

    if spectrum_bin_edges is not None:
        spectrum = rebin_uniform(spectrum, spectrum_bin_edges, bin_edges_rebin)

    return spectrum, [spectrum_bin_edges]


def hdi(
    sample: np.ndarray,
    hdi_prob: Union[float, np.number] = np.math.erf(np.sqrt(0.5)),
) -> Tuple[float, float]:
    """
    Calculates highest density interval (HDI) of sample.

    :param sample: Sample to calculate hdi for.
    :param hdi_prob:
        The highest density interval is calculated for this probiblity.
    :return:
        - Lower edge of highest density interval.
        - Upper edge of highest density interval.
    """
    sample.sort(axis=0)
    interval = int(np.ceil(len(sample) * (1 - hdi_prob)))
    candidates = sample[-interval:] - sample[:interval]
    best = candidates.argmin()
    return (sample[best], sample[best - interval])


def reduce_binning(
    bin_edges: np.ndarray,
    binning_factor: int,
    left: float = 0.0,
    right: float = np.inf,
) -> Tuple[int, int]:
    """
    Crops bin_edges to left and right and reduces the binning by
    binning_factor. Both left and right are included in the resulting axis.

    :param bin_edges: bin_edges array to work on.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.

    :return:
        - Index of original bin_edges for lowest bin edge of resulting axis.
        - Index of original bin_edges for highest bin edge of resulting axis.
    """
    left_bin = np.argmax(left < bin_edges[1:])
    right_bin = left_bin + (
        np.argmax(right <= bin_edges[left_bin::binning_factor]) * binning_factor
        or bin_edges.shape[0] - left_bin - 1
    )
    return left_bin, right_bin


def rebin_hist(
    hist: np.ndarray,
    binning_factor: int,
    bin_edges: Optional[np.ndarray] = None,
    left: float = -np.inf,
    right: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebins hist with dimension $N^M$.

    The binning is reduced by a factor of bin_width, i.e. neighboring
    bins are summed.

    :param hist: Input matrix of type $N^M$ (N bins, M dimensions).
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param bin_edges:
        Optional bin_edges array of hist, defaults to ``[0., 1., ...]``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.

    :return:
        - Rebinned matrix
        - Resulting bin edges
    """
    if bin_edges is None:
        bin_edges = np.arange(0.0, hist.shape[0] + 1)
    left_bin, right_bin = reduce_binning(bin_edges, binning_factor, left, right)
    num_dim = hist.ndim
    num_bins = (right_bin - left_bin) // binning_factor

    if not (np.array(hist.shape)[1:] == np.array(hist.shape)[:-1]).all():
        raise ValueError("Input histogram has to be square.")

    res = (
        hist[(slice(left_bin, right_bin),) * num_dim]
        .reshape(*[num_bins, binning_factor] * num_dim)
        .sum(axis=(num_dim * 2 - 1))
    )
    for i in range(1, num_dim):
        res = res.sum(axis=i)
    return res, bin_edges[left_bin : right_bin + 1 : binning_factor]


def rebin_uniform(
    hist: np.ndarray, bin_edges: np.ndarray, bin_edges_new: np.ndarray
) -> np.ndarray:
    """
    Rebins hist from binning bin_edges to bin_edges_new.

    Each count in the original histogram is randomly placed within
    the limits of the corresponding bin following a uniform probability
    distribution. A new histogram with bin edges bin_edges_new is
    filled with the resulting data.

    :param hist: Original histogram to rebin.
    :param bin_edges: bin edges of original histogram.
    :param bin_edges_new: bin edges of rebinned histogram.

    :return: Rebinned histogram.
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
    hist_norm: Optional[str] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Loads detector response matrix from file.

    :param path: Path of container file containing response matrix.
    :param hist_rema : Name of histogram for response matrix.
    :param hist_norm:
        Name of histogram used for normalization (e.g. containing
        number of simulated particles).

    :return:
        - Response matrix
        - List of bin_edges arrays
    """
    rema, bin_edges = read_spectrum(path, hist_rema)
    if (
        not (2 >= len(bin_edges) > 0)
        or len(rema.shape) != 2
        or rema.shape[0] != rema.shape[1]
    ):
        raise ValueError("Wrong response matrix dimension")
    if hist_norm:
        norm, (norm_bin_edges, *_) = read_spectrum(path, hist_norm)
        if not (
            bin_edges[0].shape == norm_bin_edges.shape
            and np.isclose(bin_edges[0], norm_bin_edges).all()
        ):
            raise ValueError(
                "Binning of response matrix and normalization histogram not equal"
            )
        rema /= norm
    return rema, bin_edges


def get_rema(
    path: Path,
    hist_rema: str,
    binning_factor: int,
    left: int,
    right: int,
    hist_norm: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtains the response matrix from container file, optionally applying
    a normalization. Crops response matrix to ``left``and ``right`` and
    does rebinning by ``binning_factor``.

    :param path: Path of container file.
    :param hist_rema: Name of detector respone matrix histogram.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.
    :param hist_norm:
        Divide detector response matrix by this histogram, e. g.
        to scale the matrix by the number of simulated particles

    :return:
        - Response matrix
        - Bin edges
    """
    rema, bin_edges = load_rema(path, hist_rema, hist_norm)
    bin_edges = bin_edges[0]

    rema_re = rebin_hist(rema, binning_factor, bin_edges, left, right)
    return rema_re


@dataclass
class SimInfo:
    """Simulation info metadata"""

    path: Path
    energy: float
    nevents: int

    @classmethod
    def from_dat_file_line(cls, line: str, sim_root: Path) -> SimInfo:
        """
        Creates SimInfo object from dat_file line

        :param line:
            Single line of dat file in format
            ``<histogram>: <energy> <number of events``.
        :param sim_root:
            Root of simulation directory. Paths are given
            relative to this directory.
        """
        path, energy, nevents = line.rsplit(maxsplit=2)
        return cls(
            sim_root / path.rstrip(":").strip().strip('"'),
            float(energy),
            int(nevents),
        )

    def __str__(self):
        """Convert to dat_file line."""
        return f"{self.path}: {self.energy} {self.nevents}"


def read_dat_file(
    dat_file_path: Path, sim_root: Optional[Path] = None
) -> List[SimInfo]:
    """Reads and parses datfile.

    :param dat_file_path: Path to datfile.
    :param sim_root:
        Optional, root of simulation directory. Paths in
        dat_file_path are given relative to this directory.
        If ``None``, it is assumed that they are given relative
        to ``dat_file_path``.

    :return: List of SimInfo objects
    """
    simulations = []
    sim_root = sim_root or dat_file_path.parents[0]
    with open(dat_file_path) as f:
        for line in f:
            simulations.append(SimInfo.from_dat_file_line(line, sim_root))
    return simulations


class SimSpec(SimInfo):
    """Simulation spectrum with metadata."""

    def __init__(
        self,
        path: Path,
        detector: Optional[str],
        energy: float,
        nevents: int,
        scale: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__(path, energy, nevents)
        self.detector = detector
        self.orig_spec, (self.bin_edges,) = read_spectrum(self.path, detector)
        self.bin_edges *= scale
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        if normalize:
            self.spec = self.orig_spec / nevents
        else:
            self.spec = self.orig_spec

    def binning_convention(self) -> float:
        """
        Determines relative shift of particle energy in comparison to bin width.
        :return:
            Binning convention. For tv-convention, this is 0.5,
            for root-convention, it is 0.0.
        """
        ebin = self.find_bin(self.energy)
        return (self.energy - self.bin_edges[ebin]) / (
            self.bin_edges[ebin + 1] - self.bin_edges[ebin]
        )

    def find_bin(self, energy: float) -> int:
        """
        Finds bin number for bin containing ``energy``.

        :param energy: Find bin containing this energy
        :return: Bin number
        """
        return np.digitize(energy, self.bin_edges) - 1


def interpolate_grid(
    grid: np.ndarray, point: float
) -> List[Tuple[int, float, float]]:
    """
    Creates a linear interpolation to ̀ `point`` given a 1D-grid.
    Finds the two closest grid points and assigns weights corresponding
    to the distance to the point. Uses only one grid point if ``point``
    is outside the grid.

    :param grid: 1D-array containing grid points
    :param point: Interpolate to this point

    :return:
        List of (index, gridpoint, weight)
    """
    digit = np.digitize(point, grid)
    if digit == 0:
        return [(digit, grid[digit], 1)]
    if digit == len(grid):
        return [(digit - 1, grid[digit - 1], 1)]
    points = grid[digit - 1], grid[digit]
    weight = (point - points[0]) / (points[1] - points[0])
    return [(digit - 1, points[0], 1 - weight), (digit, points[1], weight)]


def create_matrix(
    simulations: List[SimInfo],
    detector: Optional[str],
    max_energy: Optional[float] = None,
    scale_hist_axis: float = 1e3,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates detector response matrix.

    :param simulations:
        List of simulations used to create the detector response matrix.
    :param detector:
        Name of detector.
    :param max_energy:
        Limit maximum energy of detector response matrix. If ``None``,
        use the maximum simulated energy.
    :param scale_hist_axis:
        Scale energy axis of simulations with this parameter, for example,
        to convert MeV to keV.
    :param normalize:
        Divide matrix by number of simulated particles.
    :return: Detector response matrix and bin edges for both axes.
    """

    specs = {
        sim.energy: SimSpec(
            sim.path,
            detector,
            sim.energy,
            sim.nevents,
            scale_hist_axis,
            normalize,
        )
        for sim in simulations
    }

    energies = np.array(sorted(list(specs.keys())))
    bin_edges = specs[energies[-1]].bin_edges
    max_energy = max_energy or energies[-1]
    max_idx = np.argmin(bin_edges < max_energy)
    bin_edges = bin_edges[: max_idx + 1]
    bin_width = bin_edges[-1] - bin_edges[-2]
    rel_peak_pos = next(iter(specs.values())).binning_convention()

    # TODO: Maybe some indices are wrong here?
    mat = np.zeros(
        shape=(bin_edges.shape[0] - 1, bin_edges.shape[0] - 1), dtype=np.float64
    )
    for i, energy in enumerate(bin_edges[:-1]):
        sim_energy = energy + rel_peak_pos * bin_width
        for _, jenergy, weight in interpolate_grid(energies, sim_energy):
            shift = int(np.round(sim_energy - jenergy))
            if shift < 0:
                mat[i, 0 : i + 1] += (
                    weight
                    * specs[jenergy].spec[abs(shift) : abs(shift) + i + 1]
                )
                # mat[0:i, i] += weight * specs[jenergy].spec[abs(shift):abs(shift) + i].T
            elif shift > 0:
                mat[i, shift : i + 1] += (
                    weight * specs[jenergy].spec[0 : i - shift + 1]
                )
                # mat[shift:i, i] += weight * specs[jenergy].spec[0: i - shift].T
            else:
                mat[i, 0 : i + 1] += weight * specs[jenergy].spec[0 : i + 1]
                # mat[0:i, i] += weight * specs[jenergy].spec[0:i].T
    return mat, bin_edges, bin_edges
