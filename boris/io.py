#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2020–2023 O. Papst.
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

"""Input and output of histograms/spectra in various formats."""

from __future__ import annotations

import itertools
import warnings
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import hist

logger = logging.getLogger(__name__)


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    Convert bin centers to bin edges

    Assume that the edge is located at the midpoint between neighboring
    bin centers. The first and last bin edge are extrapolated each.

    :param centers: Bin center array of size N ≥ 3
    :return: Bin edges array of size N + 1
    """
    edges = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate(
        [
            [2.0 * edges[0] - edges[1]],
            edges,
            [2.0 * edges[-1] - edges[-2]],
        ]
    )


def get_filetype(path: Path) -> str | None:
    """
    Determines file format of path using magic bytes.
    Supports root-files, hdf5 and non-empty zip-files (used by npz).

    :param path: Path of file.
    :return: Mimetype of file or ``None``, if not sure.
    """
    with open(path, "rb") as f:
        header = f.read(4)
        signature = {
            b"PK\x03\x04": "application/zip",
            b"root": "application/root",
            b"\x89HDF": "application/x-hdf5",
            b"CDF\x01": "application/x-netcdf",
        }.get(header, None)
        if signature:
            return signature
        header += f.readline()
        line = header.split(b"\n")[0].strip()
        while line[0:1] == b"#":
            line = f.readline().strip()
        if b"\t" in line:
            return "text/tab-separated-values"
        if b"," in line:
            return "text/comma-separated-values"
        if b" " in line:
            return "text/space-separated-values"
        if line[0:1] in b"+-.0123456789":
            return "text/plain"
        return None


def get_calibration_from_file(calfile: Path, spectrum: str) -> np.ndarray:
    """
    Load energy calibration for `spectrum` from `calfile`.

    :param calfile: Calibration file, consisting of a list of filenames with
        calibration polynomials separated by a colon
    :param spectrum: Name of the spectrum to calibrate
    :raises KeyError: If no calibration for spectrum is found
    :return: Calibration polynomial
    """
    pat = re.compile(r"(.+):(\s[+\-\d\.eE]+)+$")
    with open(calfile) as f:
        for line in f:
            if res := pat.match(line.strip()):
                if res.groups()[0] == spectrum:
                    return np.array(
                        [float(c) for c in line.split(":")[-1].split()]
                    )
    raise KeyError(f"No calibration for {spectrum} found in {calfile}")


def obj_resolve_name(
    mapping: Mapping[str, Any], name: str | None = None
) -> str:
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
        return name

    keys = [
        k
        for k in mapping.keys()
        if not any(
            [
                k.endswith(suffix)
                for suffix in ["nbins", "start", "stop", "edges", "centers"]
            ]
        )
    ]
    if len(keys) == 1:
        return keys[0]
    raise KeyError("Please provide name of object")


def get_keys_in_container(path: Path) -> list[str]:
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
    elif match := re.match(
        "^text/(comma|tab|space)-separated-values$", filetype
    ):
        delimiter = {
            "comma": ",",
            "tab": "\t",
            "space": " ",
        }.get(match.groups()[0])
        return list(
            np.genfromtxt(
                path, delimiter=delimiter, names=True, max_rows=1
            ).dtype.names
        )
    return []


def read_spectrum_root(
    container: Path,
    key: str | None = None,
) -> hist.BaseHist:
    """
    Read spectrum from ROOT file

    :param container: ROOT file
    :param key: Path of spectrum in `container`
    :return: Spectrum
    """
    import uproot

    with uproot.open(container) as specfile:
        if not key:
            spectra_keys = list(specfile.keys())
            if len(spectra_keys) != 1:
                raise KeyError(
                    "Name of spectrum not given, and multiple candidates found."
                )
            key = spectra_keys[0]
        return specfile[key].to_hist()


def read_spectrum_hdf5(
    container: Path,
    key: str | None = None,
) -> hist.BaseHist:
    """
    Read spectrum for hdf5 file

    Only matrices with same axes along all dimensions supported.

    :param container: hdf5 file
    :param key: Path of spectrum in `container`
    :return: Spectrum
    """
    import h5py

    h = hist.Hist.new
    with h5py.File(container, "r") as specfile:
        key = obj_resolve_name(specfile, key)
        obj = specfile[key]
        for _ in range(obj[()].ndim):
            try:
                h = h.Regular(
                    obj.attrs["nbins"], obj.attrs["start"], obj.attrs["stop"]
                )
            except KeyError:
                try:
                    h = h.Variable(obj.attrs["edges"])
                except KeyError:
                    h = h.Integer(0, obj[()].shape[0])
        if obj[()].dtype == int:
            return h.Int64(data=obj[()])
        return h.Double(data=obj[()])


def read_spectrum_npz(
    container: Path,
    key: str | None = None,
) -> hist.BaseHist:
    """
    Read spectrum for hdf5 file

    Only matrices with same axes along all dimensions supported.

    :param container: hdf5 file
    :param key: Path of spectrum in `container`
    :return: Spectrum
    """
    regular_info = ["nbins", "start", "stop"]
    h = hist.Hist.new
    with np.load(container) as specfile:
        key = obj_resolve_name(specfile, key)
        obj = specfile[key]
        for _ in range(obj.ndim):
            if all(f"{key}_{suf}" in specfile for suf in regular_info):
                h = h.Regular(
                    *[specfile[f"{key}_{suf}"] for suf in regular_info]
                )
            elif all(s in specfile for s in ["nbins", "start", "stop"]):
                h = h.Regular(*[specfile[suf] for suf in regular_info])
            elif "edges" in specfile:
                h = h.Variable(specfile["edges"])
            elif f"{key}_edges" in specfile:
                h = h.Variable(specfile[f"{key}_edges"])
            else:
                h = h.Integer(0, obj.shape[0])
        return h.Int64(data=obj) if obj.dtype == int else h.Double(data=obj)


def read_spectrum_txt(
    container: Path,
) -> hist.BaseHist:
    """
    Read spectrum from plain text file (without headings)

    Bin edges are determined automatically either as ascending integers
    or from columns containing bin edges or bin centers. If multiple
    columns containing data are found, the first is returned.

    :param container: txt file
    :return: Spectrum
    """
    arr = np.loadtxt(container)
    if arr.ndim not in [1, 2]:
        raise ValueError("Array is wrong dimension, cannot extract bin edges.")
    elif arr.ndim == 1 or arr.shape[0] == 1:
        obj = arr.ravel()
        h = hist.Hist.new.Integer(0, obj.shape[0])
    else:
        # Transpose if spectrum given in row format
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T

        cols_asc = [
            i for i in range(arr.shape[1]) if (np.diff(arr[:, i]) > 0).all()
        ]

        if not cols_asc:
            # No binning information, assume first column is spectrum
            obj = arr[:, 0]
            h = hist.Hist.new.Integer(0, obj.shape[0])
        elif len(cols_asc) == 1:
            # Single ascending column, assume bin center information
            idx = 0 if cols_asc[0] == 1 else 1
            obj = arr[:, idx]
            centers = arr[cols_asc[0]]
            h = hist.Hist.new.Variable(centers_to_edges(centers))
        else:
            # Find bin edges columns
            for col0, col1 in itertools.permutations(cols_asc):
                if (arr[:-1, col0] == arr[1:, col1]).all():
                    # Assume first non-ascending column  is spectrum
                    for i in range(arr.shape[1]):
                        if i not in cols_asc:
                            idx = i
                            break
                    else:
                        raise RuntimeError("Could not determine bin edges")
                    obj = arr[:, idx]
                    edges = np.concatenate([arr[:, col0], [arr[-1, col1]]])
                    h = hist.Hist.new.Variable(edges)
                    break
            else:
                raise RuntimeError("Could not determine bin edges")

    # Convert to integer if lossless
    obj_int = obj.astype(int)
    if np.isclose(obj, obj_int).all():
        return h.Int64(data=obj)
    return h.Double(data=obj)


def read_spectrum_xsv(
    container: Path,
    key: str | None = None,
    delimiter: str | None = " ",
) -> hist.BaseHist:
    """
    Read spectrum from `delimiter`-separated file

    :param container: txt file which includes a header
    :param key: Path of spectrum in `container`
    :param delimiter: Delimiter separating columns
    :return: Spectrum
    """
    specfile = np.genfromtxt(container, delimiter=delimiter, names=True)
    key = obj_resolve_name(specfile, key)
    obj = specfile[key]
    try:
        h = hist.Hist.new.Regular(
            specfile["nbins"], specfile["start"], specfile["stop"]
        )
    except KeyError:
        try:
            h = hist.Hist.new.Variable(specfile["edges"])
        except KeyError:
            h = hist.Hist.new.Integer(0, obj.shape[0])

    obj_int = obj.astype(int)
    if np.isclose(obj, obj_int).all():
        return h.Int64(data=obj)
    return h.Double(data=obj)


def read_spectrum(
    spectrum: Path,
    histname: str | None = None,
) -> hist.BaseHist:
    """
    Reads spectrum to numpy array.

    :param spectrum: Path of spectrum file.
    :param histname: Name of histogram in spectrum file to load (optional).

    :return: Spectrum
    """
    filetype = get_filetype(spectrum)
    if filetype:
        if filetype == "text/plain":
            if histname:
                raise KeyError(
                    f"Cannot read key '{spectrum}' from spectrum of type '{histname}'"
                )
            return read_spectrum_txt(spectrum)
        if filetype == "application/root":
            return read_spectrum_root(spectrum, histname)
        elif filetype == "application/zip":
            return read_spectrum_npz(spectrum, histname)
        elif filetype == "application/x-hdf5":
            return read_spectrum_hdf5(spectrum, histname)
        elif match := re.match(
            "^text/(comma|tab|space)-separated-values$", filetype
        ):
            if not histname:
                # TODO: Try to use sole hist, if it exists
                raise KeyError("No histname given for container")
            delimiter = {
                "comma": ",",
                "tab": "\t",
                "space": " ",
            }.get(match.groups()[0])
            return read_spectrum_xsv(spectrum, histname, delimiter)
    raise NotImplementedError(f"Cannot read file of type '{filetype}'.")


def write_specs_hdf5(path: Path, objs: Mapping[str, hist.BaseHist]) -> None:
    """
    Write multiple spectra to a hdf5 file.

    :param path: Output file.
    :param objs: Mapping of keys and spectra to write to file.
    """
    import h5py

    compression_args = dict(
        compression="gzip",
        compression_opts=9,
    )
    with h5py.File(path, "w") as specfile:
        for key, h in objs.items():
            if isinstance(h, hist.BaseHist):
                specfile.create_dataset(key, data=h, **compression_args)
                ax = h.axes[0]
                if isinstance(ax, hist.axis.Regular):
                    specfile[key].attrs["nbins"] = ax.size
                    specfile[key].attrs["start"] = ax.edges[0]
                    specfile[key].attrs["stop"] = ax.edges[-1]
                else:
                    specfile[key].attrs["edges"] = ax.edges
            else:
                warnings.warn(f"Cannot write {key}")


def write_specs_npz(path: Path, objs: Mapping[str, hist.BaseHist]) -> None:
    """
    Write multiple spectra to a npz file.

    :param path: Output file.
    :param objs: Mapping of keys and spectra to write to file.
    """
    specfile = {}
    for key, h in objs.items():
        specfile[key] = h.values()
        if isinstance(h, hist.BaseHist):
            specfile[key] = h
            ax = h.axes[0]
            if isinstance(ax, hist.axis.Regular):
                specfile[f"{key}_nbins"] = ax.size
                specfile[f"{key}_start"] = ax.edges[0]
                specfile[f"{key}_stop"] = ax.edges[-1]
            else:
                specfile[f"{key}_edges"] = ax.edges
        else:
            warnings.warn(f"Cannot write {key}")
    np.savez_compressed(path, **specfile)


def write_specs_root(path: Path, objs: Mapping[str, hist.BaseHist]) -> None:
    """
    Write multiple spectra to a root file.

    :param path: Output file.
    :param objs: Mapping of keys and spectra to write to file.
    """
    import uproot

    with uproot.create(path) as specfile:
        for key, obj in objs.items():
            specfile[key] = obj


def write_specs(
    path: Path,
    objs: Mapping[str, hist.BaseHist],
    force_overwrite: bool = False,
) -> None:
    """
    Write multiple spectra to a file.

    :param path: Output file.
    :param objs: Mapping of keys and spectra to write to file.
    :param force_overwrite: Overwrite path if it exists.
    """
    if not force_overwrite and path.exists():
        raise Exception(f"Error writing {path}, already exists.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if path.suffix == ".npz":
        write_specs_npz(path, objs)
    elif path.suffix == ".root":
        write_specs_root(path, objs)
    elif path.suffix == ".hdf5" or path.suffix == ".h5":
        write_specs_hdf5(path, objs)
    else:
        raise Exception(f"File format {path.suffix} not supported.")


def load_rema(
    path: Path,
    key_rema: str,
) -> hist.BaseHist:
    """
    Loads detector response matrix from file.

    :param path: Path of container file containing response matrix.
    :param hist_rema : Name of histogram for response matrix.

    :return: Response matrix
    """
    rema = read_spectrum(path, key_rema)
    if rema.ndim != 2 or not (rema.axes[0].edges == rema.axes[1].edges).all():
        raise ValueError("Wrong response matrix dimension")
    return rema


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
        path, info = line.rsplit(":", maxsplit=1)
        path = sim_root / path.rstrip(":").strip().strip('"')
        info_parts = info.strip().split()

        if len(info_parts) == 1:
            energy, nevents = info_parts[0], 1
        else:
            energy, nevents, *_ = info_parts

        return cls(
            path,
            float(energy),
            int(nevents),
        )

    def __str__(self):
        """Convert to dat_file line."""
        return f"{self.path}: {self.energy} {self.nevents}"


def read_dat_file(
    dat_file_path: Path, sim_root: Path | None = None
) -> list[SimInfo]:
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


def get_simulation_spectra(
    simulations: list[SimInfo],
    detector: str | None,
) -> dict[float, hist.BaseHist]:
    """
    From a list of simulations, load spectra for given detector

    :param simulations: List of simulations
    :param detector: Name of detector to load spectra for
    :return: Dictionary that associates spectra with energies
    """
    return {
        sim.energy: hist.Hist(
            read_spectrum(sim.path, detector), storage=hist.storage.Double()
        )
        / sim.nevents
        for sim in simulations
    }
