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

"""boris – bayesian reconstruction of incident nuclear spectra."""

import logging
import warnings

import numpy as np
import pymc as pm

from pytensor import tensor as pt
from arviz import InferenceData, waic

logger = logging.getLogger(__name__)


def theuerkauf_norm(sigma, tl):
    """
    Normalization factor for Theuerkauf peak shape
    """
    return 1 / (
        (sigma**2) / tl * pt.exp(-(tl * tl) / (2.0 * sigma**2))
        + pt.sqrt(np.pi / 2.0) * sigma * (1 + pt.erf(tl / (pt.sqrt(2.0) * sigma)))
    )


def theuerkauf(x, pos, vol, sigma, tl):
    """
    Theuerkauf peak shape model with left tail

    For numerical reasons, the parameter ``tl`` is the inverse of its usual
    definition.
    """
    tl = 1.0 / (tl * sigma)
    dx = x - pos
    norm = theuerkauf_norm(sigma, tl)
    _x = pt.switch(
        dx < -tl,
        tl / (sigma**2) * (dx + tl / 2.0),
        -dx * dx / (2.0 * sigma**2),
    )
    return vol * norm * pt.exp(_x)


def fit(
    rema: np.ndarray,
    spectrum: np.ndarray,
    bin_edges: np.ndarray,
    background: np.ndarray | None = None,
    background_scale: float = 1.0,
    rema_alt: np.ndarray | None = None,
    fit_beam: bool = False,
    regularize: bool = False,
    ndraws: int = 10000,
    **kwargs,
) -> InferenceData:
    r"""
    Generates a MCMC chain, reconstructs spectrum using response matrix.

    :param rema: Response matrix of the detector
    :param spectrum: Observed spectrum
    :param bin_edges: Bin edges of all histograms
    :param background: Background spectrum
    :param background_scale: Relative live time of background spectrum
    :param rema_alt:
        Alternative matrix that is used to create a linear combination
        of two matrices (interpolate between both matrices).
    :param fit_beam: Fit a gaussian beam profile in addition to the
        fitted incoming spectrum.
    :param regularize:
        Use exponential instead of half flat priors to suppress empty bins.
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

    with pm.Model():
        # Model parameter
        if regularize:
            incident = pm.Exponential(
                "incident", incident_normalization, shape=spectrum.shape[0]
            )
        else:
            incident = pm.HalfFlat("incident", shape=spectrum.shape[0])

        if rema_alt is None:
            rema_eff = rema
            rema_eff_diag = np.diag(rema_eff)
        else:
            interpolation = pm.Uniform("interpolation", 0, 1, shape=())
            rema_eff = (1 - interpolation) * rema + interpolation * rema_alt
            rema_eff_diag = (1 - interpolation) * np.diag(
                rema
            ) + interpolation * np.diag(rema_alt)

        # rema_eff = pm.ConstantData("rema", rema_eff)
        folded = pm.Deterministic("folded", incident @ rema_eff)
        if background is None:
            folded_plus_bg = folded
        else:
            if regularize:
                background_inc = pm.Exponential(
                    "background_incident",
                    background_normalization,
                    shape=background.shape[0],
                )
            else:
                background_inc = pm.HalfFlat(
                    "background_incident",
                    shape=background.shape[0],
                )
            folded_plus_bg = pm.Deterministic(
                "folded_plus_bg", folded + background_scale * background_inc
            )

        pm.Deterministic("incident_scaled_to_fep", incident * rema_eff_diag)

        if fit_beam:
            beam_pos = pm.Uniform("beam_pos", bin_edges[0], bin_edges[-1], shape=1)
            beam_width = pm.Uniform(
                "beam_width",
                np.diff(bin_edges).min(),
                bin_edges[-1] - bin_edges[0],
                shape=1,
            )
            beam_vol = pm.HalfFlat("beam_vol", shape=1)

            # L1 regularization
            beam_tl = pm.Bound(
                "beam_tl", pm.Laplace.dist(mu=0.0, b=1.0), lower=0.0, shape=1
            )

            beam_incident = pm.Deterministic(
                "beam_incident",
                theuerkauf(
                    0.5 * (bin_edges[:-1] + bin_edges[1:]),
                    beam_pos,
                    beam_vol,
                    beam_width,
                    beam_tl,
                ),
            )
            # beam_folded = pm.Deterministic("beam_folded", beam_incident @ rema_eff)
            beam_folded = pm.Deterministic(
                "beam_folded", pt.dot(beam_incident, rema_eff)
            )
            beam_folded_plus_bg = (
                pm.Deterministic(
                    "beam_folded_plus_bg",
                    beam_folded + background_scale * background_inc,
                )
                if background is not None
                else beam_folded
            )

            pm.Deterministic(
                "beam_incident_scaled_to_fep", beam_incident * rema_eff_diag
            )

        # Measured data
        if background is not None:
            pm.Poisson(
                "background",
                background_inc,
                observed=background,
            )

        pm.Poisson("spectrum", folded_plus_bg, observed=spectrum)
        if fit_beam:
            pm.Poisson("beam_spectrum", beam_folded_plus_bg, observed=spectrum)

        step = pm.NUTS()
        start = {"incident": incident_start}
        if background is not None:
            start["background_incident"] = background_start

        import importlib.util

        numpyro_spec = importlib.util.find_spec("numpyro")
        nuts_sampler = "pymc" if numpyro_spec is None else "numpyro"

        trace = pm.sample(
            ndraws,
            step=step,
            start=start,
            idata_kwargs=dict(log_likelihood=True),
            nuts_sampler=nuts_sampler,
            **kwargs,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.add_groups({"constant_data": {"bin_edges": bin_edges}})

    for var_name in trace.observed_data.keys():
        print(f"Calculate WAIC for {var_name}:")
        print(waic(trace, var_name=var_name))
    trace.stack(sample=["chain", "draw"], inplace=True)

    return trace
