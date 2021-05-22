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
from arviz import InferenceData


def deconvolute(
    rema: np.ndarray,
    spectrum: np.ndarray,
    background: Optional[np.ndarray] = None,
    background_scale: float = 1.0,
    rema_alt: Optional[np.ndarray] = None,
    ndraws: int = 10000,
    **kwargs,
) -> InferenceData:
    """
    Generates a MCMC chain, deconvolutes spectrum using response matrix.

    :param rema: Response matrix of the detector
    :param spectrum: Observed spectrum
    :param background: Background spectrum
    :param background_scale: Relative live time of background spectrum
    :param rema_alt:
        Alternative matrix that is used to create a linear combination
        of two matrices (interpolate between both matrices).
    :param ndraws: Number of draws to sample
    :param \**kwargs: Keyword arguments are passed to ``PyMC3.sample``.

    :return: Thinned and burned MCMC trace.
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
        if rema_alt is None:
            rema_eff = rema
            rema_eff_diag = np.diag(rema)
        else:
            interpolation = pm.Uniform("interpolation", 0, 1, shape=())
            rema_eff = (1 - interpolation) * rema + interpolation * rema_alt
            rema_eff_diag = (
                (1 - interpolation) * np.diag(rema)
                + interpolation * np.diag(rema_alt)
            )
        folded = pm.Deterministic("folded", incident @ rema_eff)
        if background is None:
            folded_plus_bg = folded
        else:
            background_inc = pm.Exponential(
                "background_incident",
                background_normalization,
                shape=background.shape[0],
            )
            folded_plus_bg = pm.Deterministic(
                "folded_plus_bg", folded + background_scale * background_inc
            )

        incident_scaled_to_fep = pm.Deterministic(
            "incident_scaled_to_fep", incident * rema_eff_diag
        )

        # Measured data
        if background is not None:
            background_obs = pm.Poisson(
                "background_obs",
                background_inc,
                observed=background,
            )
        observation = pm.Poisson(
            "spectrum_obs", folded_plus_bg, observed=spectrum
        )

        step = pm.NUTS()
        start = {"incident": incident_start}
        if background is not None:
            start["background_incident"] = background_start
        trace = pm.sample(
            ndraws, step=step, start=start, return_inferencedata=True, **kwargs
        )
    return trace
