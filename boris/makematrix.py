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

"""makematrix utility for detector response matrix creation"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from boris.utils import read_spectrum, get_keys_in_container, write_hists


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
        path, energy, nevents = line.split(" ")
        return cls(sim_root / path.rstrip(":"), float(energy), int(nevents))

    def to_dat_file_line(self):
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
    """Simulation spectrum with metadata"""

    def __init__(
        self, path, detector, energy, nevents, scale=1.0, normalize=True
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

    def find_bin(self, energy) -> int:
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
    detector: str,
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
        for j, jenergy, weight in interpolate_grid(energies, sim_energy):
            shift = int(np.round(sim_energy - jenergy))
            if shift < 0:
                mat[i, 0:i] += (
                    weight * specs[jenergy].spec[abs(shift) : abs(shift) + i]
                )
                # mat[0:i, i] += weight * specs[jenergy].spec[abs(shift):abs(shift) + i].T
            elif shift > 0:
                mat[i, shift:i] += weight * specs[jenergy].spec[0 : i - shift]
                # mat[shift:i, i] += weight * specs[jenergy].spec[0: i - shift].T
            else:
                mat[i, 0:i] += weight * specs[jenergy].spec[0:i]
                # mat[0:i, i] += weight * specs[jenergy].spec[0:i].T
    return mat, bin_edges, bin_edges


def make_matrix(
    dat_file_path: Path,
    output_path: Path,
    dets: Optional[List[str]] = None,
    max_energy: Optional[float] = None,
    scale_hist_axis: float = 1e3,
    sim_dir: Optional[Path] = None,
) -> None:
    """
    Makes and writes matrix by reading dat file and simulation histograms.

    :param dat_file_path: Path to datfile.
    :param output_path: Path of created detector response matrix file.
    :param dets: List of detectors to create detector response matrices for.
        If ``None``, detector response matrices are created for all
        found detectors.
    :param max_energy:
        Limit maximum energy of detector response matrix. If ``None``,
        use the maximum simulated energy.
    :param scale_hist_axis:
        Scale energy axis of simulations with this parameter, for example,
        to convert MeV to keV.
    :param sim_dir:
        Root of simulation directory. Paths in ``dat_file_path`` are
        given relative to this directory. If ``None``, it is assumed
        that they are given relative to ``dat_file_path``.
    """
    simulations = read_dat_file(dat_file_path, sim_dir)
    dets = dets or get_keys_in_container(simulations[0].path)
    remas = {
        det: create_matrix(simulations, det, max_energy, scale_hist_axis)
        for det in dets or [None]
    }
    idx = next(iter(remas))
    write_hists(
        {det: rema[0] for det, rema in remas.items()},
        [remas[idx][1], remas[idx][2]],
        output_path,
    )


def parse_args(args: List[str]):
    """Parse makematrix command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim-dir",
        help="simulation file names are given relative to this directory (default: Directory containing datfile)",
        type=Path,
    )
    parser.add_argument(
        "--scale-hist-axis",
        help="Scale energy axis of histograms in case a different unit is used by the simulation (default: 1.0)",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--detector",
        nargs="*",
        help="Names of histograms to create response matrices for (default: All available histograms)",
    )
    parser.add_argument(
        "--max-energy",
        nargs="?",
        help="Maximum energy of created response matrix",
        type=float,
    )
    parser.add_argument(
        "datfile",
        help=(
            "datfile containing simulation information, each line has format "
            "`<simulation_hists.root>: <energy> <number of particles`"
        ),
        type=Path,
    )
    parser.add_argument(
        "output_path",
        help="Write resulting response matrix to this file.",
        type=Path,
    )
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    make_matrix(
        args.datfile,
        args.output_path,
        args.detector,
        args.max_energy,
        args.scale_hist_axis,
        args.sim_dir,
    )


def init():
    if __name__ == "__main__":
        main()


init()
