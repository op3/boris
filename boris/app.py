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

"""boris command line interface."""

import contextlib
import logging
from pathlib import Path
from typing import Any, Callable, Generator, List, Mapping, Optional

import numpy as np

from boris.utils import (
    create_matrix,
    get_keys_in_container,
    get_rema,
    hdi,
    read_dat_file,
    read_rebin_spectrum,
    read_spectrum,
    write_hist,
    write_hists,
)

logger = logging.getLogger(__name__)


def setup_logging():
    """Prepares logger, sets message format."""
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


@contextlib.contextmanager
def do_step(text: str, simple: bool = False) -> Generator[None, None, None]:
    """
    Contextmanager to print helpful progress messages

    :param text: Task that is going to be executed.
    :param simple:
        If True, execution message is logged only after execution completed,
        otherwise, it is displayed before and after execution.
    """
    if not simple:
        logger.info(f"{text} ...")
    try:
        yield
        if simple:
            logger.info(text)
        else:
            logger.info(f"{text} complete")
    except BaseException as e:
        logger.error(f"{text} failed:")
        logger.error(e, exc_info=True)
        exit(1)


def check_if_exists(path: Path):
    """
    checks if path already exists

    :param path: Path to check for
    :raises Exception: if path exists
    """
    if path.exists():
        raise Exception(f"Path {path} already exists")


def boris(
    matrix: Path,
    observed_spectrum: Path,
    incident_spectrum: Path,
    binning_factor: int,
    left: int,
    right: int,
    histname: Optional[str] = None,
    rema_name: str = "rema",
    background_spectrum: Optional[Path] = None,
    background_name: Optional[str] = None,
    background_scale: float = 1.0,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
    norm_hist: Optional[str] = None,
    deconvolute: Optional[Callable[..., Mapping]] = None,
    **kwargs: Any,
) -> None:
    r"""
    Loads response matrix and spectrum, samples MCMC chain, writes output.

    :param matrix: Path of container file containing the response matrix.
    :param observed_spectrum:
        Path of container file containing the observed spectrum.
    :param incident_spectrum:
        Write MCMC chain trace of incident spectrum to this file.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.
    :param histname:
        Name of histogram in observed_spectrum to read (optional)
    :param rema_name:
        Name of the detector response matrix in matrix file
        (only required if not unique).
    :param background_spectrum:
        Path of container file containing background spectrum.
    :param background_name:
        Name of background spectrum, loaded from ``observed_spectrum``
        or (if given) ``background_spectrum``.
    :param background_scale:
        Relative scale of background spectrum to observed spectrum
        (e. g. ratio of live times).
    :param cal_bin_centers:
        Optional energy calibration polynomial that is used to
        calibrate the energy of the bin_centers of the observed
        spectrum and background spectrum. (hdtv-style calibration)
    :param cal_bin_edges:
        Optional energy calibration polynomial that is used to
        calibrate the energy of the bin_edges of the observed
        spectrum and background spectrum. (root-style calibration)
    :param norm_hist:
        Divide detector response matrix by this histogram
        (e. g., to correct for number of simulated particles).
    :param deconvolute: Alternate function used for deconvolution.
    :param \**kwargs:
        Keyword arguments are passed to ``deconvolute`` function.
    """
    check_if_exists(incident_spectrum)
    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema, rema_bin_edges = get_rema(
            matrix, rema_name, binning_factor, left, right, norm_hist
        )

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading observed spectrum {observed_spectrum}{print_histname}"
    ):
        spectrum, _ = read_rebin_spectrum(
            observed_spectrum,
            rema_bin_edges,
            histname,
            cal_bin_centers,
            cal_bin_edges,
        )

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            background, _ = read_rebin_spectrum(
                background_spectrum or observed_spectrum,
                rema_bin_edges,
                background_name,
                cal_bin_centers,
                cal_bin_edges,
            )

    with do_step("🎲 Sampling from posterior distribution"):
        if deconvolute is None:
            from boris.core import deconvolute
        trace = deconvolute(
            rema,
            spectrum,
            background,
            background_scale,
            **kwargs,
        )

    with do_step(f"💾 Writing incident spectrum trace to {incident_spectrum}"):
        write_hist(
            incident_spectrum, "incident", trace["incident"], rema_bin_edges
        )


def sirob(
    matrix: Path,
    incident_spectrum: Path,
    observed_spectrum: Path,
    binning_factor: int,
    left: int,
    right: int,
    histname: Optional[str] = None,
    rema_name: str = "rema",
    background_spectrum: Optional[Path] = None,
    background_name: Optional[str] = None,
    background_scale: float = 1.0,
    cal_bin_centers: Optional[List[float]] = None,
    cal_bin_edges: Optional[List[float]] = None,
    norm_hist: Optional[str] = None,
) -> None:
    """
    Performs convolution of incident spectrum with detector response matrix
    to reproduce the incident spectrum, optionally with an additional
    background contribution.

    :param matrix: Path of container file containing the response matrix.
    :param incident_spectrum:
        Path of container file containing the incident spectrum.
    :param observed_spectrum: Write resulting observed spectrum to this file.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.
    :param histname:
        Name of histogram in incident_spectrum to read (optional).
    :param rema_name:
        Name of the detector response matrix in matrix file
        (only required if not unique).
    :param background_spectrum:
        Path of container file containing background spectrum.
    :param background_name:
        Name of background spectrum, loaded from ``incident_spectrum``
        or (if given) ``background_spectrum``.
    :param background_scale:
        Relative scale of background spectrum to incident spectrum
        (e. g. ratio of live times).
    :param cal_bin_centers:
        Optional energy calibration polynomial that is used to
        calibrate the energy of the bin_centers of the incident
        spectrum and background spectrum. (hdtv-style calibration)
    :param cal_bin_edges:
        Optional energy calibration polynomial that is used to
        calibrate the energy of the bin_edges of the incident
        spectrum and background spectrum. (root-style calibration)
    :param norm_hist: Divide detector response matrix by this histogram
        (e. g., to correct for number of simulated particles).
    :param deconvolute: Function used for deconvolution.
    :param kwargs: Passed to ``deconvolute`` function.
    """
    check_if_exists(observed_spectrum)
    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema, rema_bin_edges = get_rema(
            matrix, rema_name, binning_factor, left, right, norm_hist
        )

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading incident spectrum {incident_spectrum}{print_histname}"
    ):
        incident, (spectrum_bin_edges,) = read_rebin_spectrum(
            incident_spectrum,
            rema_bin_edges,
            histname,
            cal_bin_centers,
            cal_bin_edges,
        )

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            background, _ = read_rebin_spectrum(
                background_spectrum or incident_spectrum,
                rema_bin_edges,
                background_name,
                cal_bin_centers,
                cal_bin_edges,
            )

    with do_step("Calculating observed (convoluted) spectrum"):
        observed = incident @ rema
        if background is not None:
            observed += background_scale * background

    with do_step(f"Writing observed spectrum to {observed_spectrum}"):
        write_hist(observed_spectrum, "observed", observed, spectrum_bin_edges)


def boris2spec(
    incident_spectrum: Path,
    output_path: Optional[Path] = None,
    plot: bool = False,
    get_mean: bool = False,
    get_median: bool = False,
    # get_mode: bool = False,
    get_variance: bool = False,
    get_std_dev: bool = False,
    get_min: bool = False,
    get_max: bool = False,
    get_hdi: bool = False,
    hdi_prob: float = np.math.erf(np.sqrt(0.5)),
) -> None:
    """
    Creates and/or plots spectra from boris trace file.

    :param incident_spectrum:
        Path of container file containing incident spectrum trace
        generated by ``boris``.
    :param output_path:
        Path of container file which is created containing the generated
        spectra (optional).
    :param plot: Display matplotlib window of all spectra (optional).
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
    if output_path:
        check_if_exists(output_path)
    spec, bin_edges = read_spectrum(incident_spectrum, "incident")
    bin_edges = bin_edges[-1]

    res = {}

    if get_mean:
        res["mean"] = np.mean(spec, axis=0)

    if get_median:
        res["median"] = np.median(spec, axis=0)

    # if get_mode:
    #    raise NotImplementedError("mode not yet implemented")
    #    #res["mode"] = get_mode(spec, axis=0)

    if get_variance:
        res["var"] = np.var(spec, axis=0)

    if get_std_dev:
        res["std"] = np.std(spec, axis=0)

    if get_min:
        res["min"] = np.min(spec, axis=0)

    if get_max:
        res["max"] = np.max(spec, axis=0)

    if get_hdi:
        res["hdi_lo"], res["hdi_hi"] = hdi(spec, hdi_prob=hdi_prob)

    if plot:
        import matplotlib.pyplot as plt

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if get_hdi:
            plt.fill_between(
                bin_centers,
                res["hdi_lo"],
                res["hdi_hi"],
                label="Highest Density Interval",
                step="mid",
                alpha=0.3,
            )

        label = {
            "mean": "Mean",
            "median": "Median",
            "mode": "Mode",
            "std": "Standard Deviation",
            "var": "Variance",
            "min": "Minimum",
            "max": "Maximum",
        }

        for key in label.keys():
            if key in res:
                plt.step(bin_centers, res[key], where="mid", label=label[key])

        plt.legend()
        plt.tight_layout()
        plt.show()

    if output_path:
        write_hists(res, bin_edges, output_path)


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
    check_if_exists(output_path)
    with do_step(f"Reading simulation dat file {dat_file_path}"):
        simulations = read_dat_file(dat_file_path, sim_dir)
        dets = dets or get_keys_in_container(simulations[0].path)

    remas = dict()
    for det in dets or [None]:
        with do_step(f"Creating matrix for detector {det}"):
            remas[det] = create_matrix(
                simulations, det, max_energy, scale_hist_axis
            )

    with do_step(f"Writing created matrices to {output_path}"):
        idx = next(iter(remas))
        write_hists(
            {det: rema[0] for det, rema in remas.items()},
            [remas[idx][1], remas[idx][2]],
            output_path,
        )


def check_matrix(
    matrix: Path,
    binning_factor: int,
    left: int,
    right: int,
    rema_name: str = "rema",
    norm_hist: Optional[str] = None,
) -> None:
    """
    Visualizes detector response matrix.

    :param matrix: Path of container file containing the response matrix.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop ``bin_edges`` of response matrix to the lowest bin
        still containing ``left``.
    :param right:
        Crop ``bin_edges`` of response matrix to the highest bin
        still containing ``right``.
    :param rema_name:
        Name of the detector response matrix in matrix file
        (only required if not unique).
    :param norm_hist:
        Divide detector response matrix by this histogram
        (e. g., to correct for number of simulated particles).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema, rema_bin_edges = get_rema(
            matrix, rema_name, binning_factor, left, right, norm_hist
        )

    plt.pcolormesh(
        rema_bin_edges,
        rema_bin_edges,
        rema,
        norm=LogNorm(vmin=rema[rema > 1e-20].min(), vmax=rema.max()),
    )
    plt.title(rema_name)
    plt.xlabel("Observed energy")
    plt.ylabel("Particle energy")
    plt.tight_layout()
    plt.show()
