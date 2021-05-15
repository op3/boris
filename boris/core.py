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

"""boris – bayesian deconvolution of nuclear spectra."""

from typing import Optional

import numpy as np
import pymc3 as pm


def deconvolute(
    rema: np.ndarray,
    spectrum: np.ndarray,
    background: Optional[np.ndarray] = None,
    background_scale: float = 1.0,
    ndraws: int = 10000,
    thin: int = 1,
    burn: int = 1000,
    **kwargs,
) -> pm.backends.base.MultiTrace:
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
                "background_obs",
                background_inc,
                observed=background,
            )
        observation = pm.Poisson(
            "spectrum_obs", spectrum_detector, observed=spectrum
        )

        step = pm.NUTS()
        start = {"incident": incident_start}
        if background is not None:
            start["background"] = background_start
        trace = pm.sample(
            ndraws, step=step, start=start, return_inferencedata=False, **kwargs
        )
    return trace[burn::thin]
