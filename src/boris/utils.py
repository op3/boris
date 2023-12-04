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

"""Utils for input and output preparation."""

from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Mapping, Literal

import numpy as np
import numpy.typing as npt
import hist
from arviz import hdi

from boris.io import read_spectrum, load_rema

logger = logging.getLogger(__name__)

one_sigma = np.math.erf(np.sqrt(0.5))


def get_bin_edges_from_calibration(
    num_bins: int,
    calibration: np.ndarray,
    convention: Literal["edges", "centers"] = "edges",
) -> np.ndarray:
    """
    Calculate bin edges array from energy calibration

    :param num_bins: Number of bins in corresponding histogram
    :param calibration: Coefficients of calibration polynomial
    :param convention: If calibration applies to bin `edges` or bin `centers`
    :return: Bin edges array
    """
    offset = {"centers": -0.5, "edges": 0.0}[convention]
    cal = np.poly1d(calibration[::-1])
    orig = np.arange(num_bins + 1) + offset
    return cal(orig)


def read_rebin_spectrum(
    path: Path,
    bin_edges_rebin: np.ndarray,
    histname: str | None = None,
    calibration: npt.ArrayLike | None = None,
    convention: Literal["edges", "centers"] = "edges",
) -> hist.BaseHist:
    """
    Reads and rebin spectrum.

    :param spectrum: Path to file containing spectrum.
    :param bin_edges_rebin: Rebin spectrum to this bin_edges array.
    :param histname:
        Provide object name for files containing multiple objects.
    :param calibration:
        Polynomial coefficients to calibrate spectrum before rebinning.
    :param convention:
        `calibration` calibrates `"edges"` (default) or `"centers`.
    :return:
        - Histogram
        - Bin edges
    """
    spectrum = read_spectrum(path, histname)
    if spectrum.values().dtype != int:
        spectrum_new = hist.Hist(
            *spectrum.axes,
            storage=hist.storage.Int64(),
            data=spectrum.values().astype(int),
        )
        if not np.allclose(spectrum, spectrum_new):
            raise ValueError("Spectrum contents are not of type integer.")
        spectrum = spectrum_new
    if (spectrum.values() < 0).any():
        raise ValueError("Spectrum contents are not non-negative.")

    if calibration:
        if not isinstance(spectrum.axes[0], hist.axis.Integer):
            logger.warning("Ignoring binning information of spectrum")
        spectrum = hist.Hist.new.Variable(
            get_bin_edges_from_calibration(
                spectrum.shape[0], calibration, convention
            )
        ).Int64(data=spectrum.values())

    spec_rebin = rebin_uniform(
        spectrum.values(), spectrum.axes[0].edges, bin_edges_rebin
    )

    return hist.Hist.new.Variable(bin_edges_rebin).Int64(data=spec_rebin)


def rebin_uniform(
    hist_old: np.ndarray,
    bin_edges_old: np.ndarray,
    bin_edges_new: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rebin histogram `hist_old` with binning `bin_edges_old` to `bin_edges_new`,
    keeping Poisson-statistics intact by resampling with uniform distribution.

    Args:
        hist_old: 1D histogram to be rebinned
        bin_edges_old: Bin edges of existing histogram
        bin_edges_new: Bin edges of new histogram
        rng: Random number generator

    Return:
        Histogram rebinned to bin_edges_new
    """
    if not rng:
        rng = np.random.default_rng()

    if hist_old.dtype != int:
        raise ValueError("Spectrum contents are not of type integer.")

    # Expand old histogram (think of this as flow bins)
    bin_edges_old = np.concatenate(
        [[np.finfo(np.double).min], bin_edges_old, [np.finfo(np.double).max]]
    )
    hist_old = np.concatenate([[0], hist_old, [0]])

    # Create unified bin_edges_all array, idx_rev notes new index for each element of `bin_edges_old`/`bin_edges_new`.
    bin_edges_all, idx_rev = np.unique(
        np.concatenate([bin_edges_new, bin_edges_old]), return_inverse=True
    )

    # Indices to split bin_edges_all at bin_edges_old/bin_edges_new
    split_all_on_new, _, split_all_on_old, _ = np.split(
        idx_rev, [bin_edges_new.shape[0], bin_edges_new.shape[0] + 1, -1]
    )

    # Create unnormalized pval arrays for all bins, split for each bin in old histogram
    ratios = np.split(np.diff(bin_edges_all), split_all_on_old)

    # Draw number of counts for each bin in bin_edges_all
    i = 0
    hist_all = np.zeros(bin_edges_all.shape[0] - 1)
    for h, r in zip(hist_old, ratios):
        if len(r) == 1:
            hist_all[i] = h
        else:
            hist_all[i : i + len(r)] = rng.multinomial(h, r / np.sum(r))
        i += len(r)

    # Sum over neighboring bins to reduce binning to bin_edges_new
    return np.add.reduceat(hist_all, split_all_on_new)[:-1]


def rebin_uniform_continuous(
    hist_old: np.ndarray, bin_edges_old: np.ndarray, bin_edges_new: np.ndarray
) -> np.ndarray:
    """
    Rebin histogram `hist_old` with binning `bin_edges_old` to `bin_edges_new`,
    without resampling, bin contents are divided smoothly.

    Args:
        hist_old: 1D histogram to be rebinned
        bin_edges_old: Bin edges of existing histogram
        bin_edges_new: Bin edges of new histogram

    Return:
        Histogram rebinned to bin_edges_new
    """
    # Expand old histogram (think of this as flow bins)
    bin_edges_old = np.concatenate(
        [[np.finfo(np.double).min], bin_edges_old, [np.finfo(np.double).max]]
    )
    hist_old = np.concatenate([[0], hist_old, [0]])

    # Create unified bin_edges_all array, idx_rev notes new index for each element of `bin_edges_old`/`bin_edges_new`.
    bin_edges_all, idx_rev = np.unique(
        np.concatenate([bin_edges_new, bin_edges_old]), return_inverse=True
    )

    # Indices to split bin_edges_all at bin_edges_old/bin_edges_new
    split_all_on_new, _, split_all_on_old, _ = np.split(
        idx_rev, [bin_edges_new.shape[0], bin_edges_new.shape[0] + 1, -1]
    )

    # Create unnormalized pval arrays for all bins, split for each bin in old histogram
    ratios = np.split(np.diff(bin_edges_all), split_all_on_old)

    # Draw number of counts for each bin in bin_edges_all
    i = 0
    hist_all = np.zeros(bin_edges_all.shape[0] - 1)
    for h, r in zip(hist_old, ratios):
        if len(r) == 1:
            hist_all[i] = h
        else:
            hist_all[i : i + len(r)] = h * r / np.sum(r)
        i += len(r)

    # Sum over neighboring bins to reduce binning to bin_edges_new
    return np.add.reduceat(hist_all, split_all_on_new)[:-1]


def get_rema(
    path: Path,
    key_rema: str,
    binning_factor: int,
    left: int,
    right: int,
) -> hist.BaseHist:
    """
    Obtains the response matrix from container file.Crops response matrix
    to ``left``and ``right`` and does rebinning by ``binning_factor``.

    :param path: Path of container file.
    :param key_rema: Name of detector respone matrix histogram.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop response matrix to the lowest bin still containing ``left``.
    :param right:
        Crop response matrix to the highest bin not containing ``right``.

    :return: Response matrix, which is cropped and rebinned.
    """
    rema = load_rema(path, key_rema)
    return rema[
        hist.loc(left) : hist.loc(right) + 1 : hist.rebin(binning_factor),
        hist.loc(left) : hist.loc(right) + 1 : hist.rebin(binning_factor),
    ]


def interpolate_grid(
    grid: np.ndarray, point: float
) -> list[tuple[int, float, float]]:
    """
    Create a linear interpolation to ̀ `point`` given a 1D-grid

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


def shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift `arr` by `shift` number of indices, fill rest with zeros

    :param arr: Array to be shifted
    :param shift: Number of indices to shift to the left/right
    :return: Shifted array
    """
    result = np.zeros_like(arr)
    if shift > 0:
        result[shift:] = arr[:-shift]
    elif shift < 0:
        result[:shift] = arr[-shift:]
    else:
        result[:] = arr
    return result


def create_matrix(
    specs: Mapping[float, hist.BaseHist], bins: int, start: float, stop: float
) -> hist.BaseHist:
    """
    Create detector response matrix from simulated spectra for
    monoenergetic radiation

    For missing energies, a linear interpolation of neighboring
    simulations is created, shifting their energy to match the
    position of the full-energy peak. Non-triangular detector
    response matrices are not supported.

    :param specs:
        Dictionary of simulated energy and resulting spectrum.
    :param bins:
        Number of bins for response matrix axes.
    :param start:
        Lower edge of first bin of response matrix axes.
    :param stop:
        Upper edge of last bin of response matrix axes.
    :return: Detector response matrix.
    """
    axis = hist.axis.Regular(bins, start, stop, flow=False)
    res = hist.Hist(axis, axis, storage=hist.storage.Double())
    energies = np.array(sorted(list(specs.keys())))

    OffsetSpec = namedtuple("OffsetSpec", ["offset", "bin_fep", "spec"])
    specs_rebin = {}
    for energy, spec in specs.items():
        # Rebin the binning to the target axis. This might require adding an offset
        sim_bin_fep = (np.argmin(spec.axes[0].edges <= energy) - 1) % (bins + 1)
        mat_bin_fep = (np.argmin(axis.edges <= energy) - 1) % (bins + 1)
        offset = axis.value(mat_bin_fep) - spec.axes[0].value(sim_bin_fep)
        specs_rebin[energy] = OffsetSpec(
            offset=offset,
            bin_fep=mat_bin_fep,
            spec=rebin_uniform_continuous(
                specs[energy].values(),
                specs[energy].axes[0].edges + offset,
                axis.edges,
            ),
        )

    for i, energy in enumerate(axis.centers):
        for _, sim_energy, weight in interpolate_grid(energies, energy):
            offset = specs_rebin[sim_energy].offset
            bin_fep = specs_rebin[sim_energy].bin_fep
            spec = specs_rebin[sim_energy].spec
            res.values()[i, :] += weight * shift(spec, i - bin_fep)
            # specs[sim_energy].spec.axes[0].edges,
            # axis.edges - (energy - sim_energy)) # TODO: Not sure about the signs
    res.values()[:, :] = np.tril(res.values())
    return res


class QuantityExtractor:
    """Extract statistical quantities from a trace"""

    def __init__(
        self,
        *,
        mean: bool = False,
        median: bool = False,
        variance: bool = False,
        std_dev: bool = False,
        min: bool = False,
        max: bool = False,
        hdi: bool = False,
        hdi_prob: float = one_sigma,
    ):
        """
        Initialize QuantityExtractor

        :param mean: Obtain mean value of trace
        :param median: Obtain median value of trace
        :param variance: Obtain variance of trace
        :param std_dev: Obtain standard deviation of trace
        :param min: Obtain minimum of trace
        :param max: Obtain maximum of trace
        :param hdi: Obtain highest density interval of trace
        :param hdi_prob:
            Probability for which the highest density interval will be computed.
            Defaults to 1σ.
        """
        self.mean = mean
        self.median = median
        self.variance = variance
        self.std_dev = std_dev
        self.min = min
        self.max = max
        self.hdi = hdi
        self.hdi_prob = hdi_prob

    def extract(self, data, var_name: None | str = None):
        """
        Extract definied quantities form `data`.

        :param data: Trace to extract quantities from
        :param var_name: Optional
        """
        prefix = f"{var_name}_" if var_name else ""
        res = {}

        if self.mean:
            res[f"{prefix}mean"] = data.mean(axis=1)

        if self.median:
            res[f"{prefix}median"] = np.median(data, axis=1)

        if self.variance:
            res[f"{prefix}variance"] = data.var(axis=1)

        if self.std_dev:
            res[f"{prefix}std_dev"] = data.std(axis=1)

        if self.min:
            res[f"{prefix}min"] = data.min(axis=1)

        if self.max:
            res[f"{prefix}max"] = data.max(axis=1)

        if self.hdi:
            res[f"{prefix}hdi_lo"], res[f"{prefix}hdi_hi"] = hdi(
                data.T, hdi_prob=self.hdi_prob
            ).T

        return res


def get_quantities(
    trace_file: Path,
    var_name: str,
    get_mean: bool = False,
    get_median: bool = False,
    # get_mode: bool = False,
    get_variance: bool = False,
    get_std_dev: bool = False,
    get_min: bool = False,
    get_max: bool = False,
    get_hdi: bool = False,
    hdi_prob: float = one_sigma,
) -> dict[str, tuple[hist.BaseHist, float]]:
    """
    Creates and/or plots spectra from boris trace file.

    :param trace_file:
        Path of container file containing traces generated by ``boris``.
    :param var_name: Name of variable that is evaluated.
    :param get_mean: Generate spectrum containing mean of each bin.
    :param get_median: Generate spectrum containing median of each bin.
    :param get_variance: Generate spectrum containing variane of each bin.
    :param get_std_dev:
        Generate spectrum containing standard deviation of each bin.
    :param get_min: Generate spectrum containing min of each bin.
    :param get_max: Generate spectrum containing max of each bin.
    :param get_hdi:
        Generate spectra containing highest density interval of each bin
        (also known as shortest coverage interval).
    :param hdi_prob:
        Probability for which the highest density interval will be computed.
        Defaults to 1σ.
    """
    spec = read_spectrum(trace_file, var_name)

    if spec.ndim == 1:
        return {var_name: spec}

    res = {}

    if get_mean:
        res[f"{var_name}_mean"] = np.mean(spec, axis=0)

    if get_median:
        res[f"{var_name}_median"] = np.median(spec, axis=0)

    # if get_mode:
    #    raise NotImplementedError("mode not yet implemented")
    #    #res[f"{var_name}_mode"] = get_mode(spec, axis=0)

    if get_variance:
        res[f"{var_name}_var"] = np.var(spec, axis=0)

    if get_std_dev:
        res[f"{var_name}_std"] = np.std(spec, axis=0)

    if get_min:
        res[f"{var_name}_min"] = np.min(spec, axis=0)

    if get_max:
        res[f"{var_name}_max"] = np.max(spec, axis=0)

    if get_hdi:
        res[f"{var_name}_hdi_lo"], res[f"{var_name}_hdi_hi"] = hdi(
            spec[None, :, :], hdi_prob=hdi_prob
        ).T

    return res
