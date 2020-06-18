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

"""boris – bayesian deconvolution of nuclear spectra."""

from typing import Union, Tuple, Optional
from pathlib import Path

import numpy as np
import pymc3 as pm
from pymc3.backends.base import MultiTrace

import uproot


def rebin_hist(
    hist: np.ndarray, bin_width: int, left: int = 0, right: int = None
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
        rebinned matrix, resulting bin edges
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
    with uproot.open(path) as response_file:
        response = response_file["rema"]
        nsim = response_file["n_simulated_particles"]
    rema, rema_bin_edges = response.numpy()
    if not np.isclose(np.diff(rema_bin_edges), 1.0).all():
        raise NotImplementedError(
            "Response matrix with binning unequal to 1 keV not yet implemented."
        )
    rema_nsim = nsim.numpy()[0]

    # with ROOT.TFile(str(path)) as response_file:
    #    response = response_file.Get("rema")
    #    nsim = response_file.Get("n_simulated_particles")
    # rema = np.asarray(response)[1:-1]
    # rema_nsim = np.asarray(nsim)[1:-1]
    # rema_bin_edges = np.array([response.GetBinLowEdge(i)
    #                           for i in range(1, response.shape[0]+2)])

    rema_re = rebin_hist(rema, bin_width, left, right)
    rema_nsim_re = rebin_hist(rema_nsim, bin_width, left, right)
    return (rema_re[0], *rema_nsim_re)


def deconvolute(
    rema: np.ndarray,
    spectrum: np.ndarray,
    background: Optional[np.ndarray] = None,
    background_scale: float = 1.0,
    ndraws: int = 10000,
    tune: int = 500,
    thin: int = 1,
    burn: int = 1000,
    **kwargs,
) -> MultiTrace:
    """
    Generate a MCMC chain, deconvoluting spectrum using rema

    Args:
        rema: response matrix of the detector
        spectrum: observed spectrum
        background: background spectrum
        background_scale: relative live time of background spectrum
        ndraws: number of draws to sample
        tune: number of steps to tune parameters
        thin: thinning factor to decrease autocorrelation time
        burn: discard initial steps (burn-in time)
        **kwargs are passed to PyMC3.sample

    Returns:
        thinned and burned MCMC trace
    """
    background_scale = 1 / background_scale
    if background is None:
        spectrum_wobg = spectrum
    else:
        if not background.shape == spectrum.shape:
            raise ValueError("Mismatch of background and spectrum dimensions.")
        spectrum_wobg = spectrum - background_scale * background
        background_start = np.clip(background, 1, np.inf)
        background_normalization = 1 / np.mean(background)

    incident_start = np.clip(spectrum_wobg @ np.linalg.inv(rema), 1, np.inf)
    incident_normalization = 1 / np.mean(incident_start)

    with pm.Model() as model:
        # Model parameter
        incident = pm.Exponential(
            "incident", incident_normalization, shape=spectrum.shape[0]
        )
        folded = incident @ rema
        if background is None:
            spectrum_detector = folded
        else:
            background_inc = pm.Exponential(
                "background",
                background_normalization,
                shape=background.shape[0],
            )
            spectrum_detector = folded + background_scale * background_inc

        # Measured data
        if background is not None:
            background_obs = pm.Poisson(
                "background_obs", background_inc, observed=background,
            )
        observation = pm.Poisson(
            "spectrum_obs", spectrum_detector, observed=spectrum
        )

        step = pm.NUTS()
        start = {"incident": incident_start}
        if background is not None:
            start["background"] = background_start
        trace = pm.sample(ndraws, step=step, start=start, **kwargs)
    return trace[burn::thin]
