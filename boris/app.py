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
from typing import Any, Callable, Generator, List, Mapping, Optional, Literal

import hist
from tqdm import tqdm

from boris.utils import (
    create_matrix,
    get_rema,
    get_quantities,
    one_sigma,
    read_rebin_spectrum,
    QuantityExtractor,
)
from boris.io import (
    get_simulation_spectra,
    get_keys_in_container,
    read_dat_file,
    write_specs,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    """Prepares logger, sets message format."""
    logging.captureWarnings(True)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
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
    Check if path already exists

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
    calibration: List[float] | None = None,
    convention: Literal["edges", "centers"] = "edges",
    matrix_alt: Optional[Path] = None,
    force_overwrite: bool = False,
    deconvolute: Optional[Callable[..., Mapping]] = None,
    fit_beam=False,
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
    :param calibration:
        Polynomial coefficients to calibrate spectrum before rebinning.
    :param convention:
        `calibration` calibrates `"edges"` (default) or `"centers"`.
    :param matrix_alt:
        Path of container file containing alternative matrix
        that is used to create a linear combination of two
        matrices (interpolate between both matrices).
    :param force_overwrite: Overwrite ``incident_spectrum`` if it exists.
    :param deconvolute: Alternate function used for deconvolution.
    :param fit_beam: Perform fit of beam profile
    :param \**kwargs:
        Keyword arguments are passed to ``deconvolute`` function.
    """
    if not force_overwrite:
        check_if_exists(incident_spectrum)
    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema = get_rema(matrix, rema_name, binning_factor, left, right)
        edges = rema.axes[0].edges
        logger.debug(f"Bin edges: [{edges[0]}, {edges[1]}, …, {edges[-1]}]")
        logger.debug(f"Response matrix shape: {rema.shape}")
        logger.debug(f"Response matrix diagonal:\n{rema.values().diagonal()}")

    rema_alt = None
    if matrix_alt:
        with do_step(
            f"Reading alternative response matrix {rema_name} from {matrix_alt}"
        ):
            rema_alt = get_rema(
                matrix_alt, rema_name, binning_factor, left, right
            )
            logger.debug(
                f"Alternative response matrix diagonal:\n{rema_alt.values().diagonal()}"
            )

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading observed spectrum {observed_spectrum}{print_histname}"
    ):
        spectrum = read_rebin_spectrum(
            observed_spectrum,
            edges,
            histname,
            calibration,
            convention,
        )
        logger.debug(f"Observed spectrum:\n{spectrum}")

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            background = read_rebin_spectrum(
                background_spectrum or observed_spectrum,
                edges,
                background_name,
                calibration,
                convention,
            )
            print(background.values())
            logger.debug(f"Background spectrum:\n{background}")

    with do_step("🎲 Sampling from posterior distribution"):
        if deconvolute is None:
            from boris.core import deconvolute
        trace = deconvolute(
            rema.values(),
            spectrum.values(),
            rema.axes[0].edges,
            background.values() if background else None,
            background_scale,
            rema_alt.values() if rema_alt else None,
            fit_beam,
            **kwargs,
        )

    with do_step(f"💾 Writing incident spectrum trace to {incident_spectrum}"):
        import arviz

        trace.sample_stats = trace.sample_stats.reset_index("sample")
        trace.posterior = trace.posterior.reset_index("sample")
        trace.log_likelihood = trace.log_likelihood.reset_index("sample")
        trace.constant_data = trace.constant_data.reset_index("sample")
        arviz.to_netcdf(trace, incident_spectrum)


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
    calibration: List[float] | None = None,
    convention: Literal["edges", "centers"] = "edges",
    force_overwrite: bool = False,
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
    :param calibration:
        Polynomial coefficients to calibrate spectrum before rebinning.
    :param convention:
        `calibration` calibrates `"edges"` (default) or `"centers`.
    :param force_overwrite: Overwrite output_path if it exists.
    :param deconvolute: Function used for deconvolution.
    :param kwargs: Passed to ``deconvolute`` function.
    """
    if not force_overwrite:
        check_if_exists(observed_spectrum)
    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema = get_rema(matrix, rema_name, binning_factor, left, right)
        edges = rema.axes[0].edges
        logger.debug(f"Bin edges: [{edges[0]}, {edges[1]}, …, {edges[-1]}]")
        logger.debug(f"Response matrix shape: {rema.shape}")
        # logger.debug(f"Response matrix diagonal:\n{rema.values().diagonal()}")

    print_histname = f" ({histname})" if histname else ""
    with do_step(
        f"Reading incident spectrum {incident_spectrum}{print_histname}"
    ):
        incident = read_rebin_spectrum(
            incident_spectrum,
            rema.axes[0].edges,
            histname,
            calibration,
            convention,
        )
        logger.debug(f"Incident spectrum:\n{incident}")

    background = None
    if background_spectrum is not None or background_name is not None:
        print_bgname = f" ({background_name})" if histname else ""
        with do_step(
            f"Reading background spectrum {background_spectrum or observed_spectrum}{print_bgname}"
        ):
            background = read_rebin_spectrum(
                background_spectrum or incident_spectrum,
                rema.axes[0].edges,
                background_name,
                calibration,
                convention,
            )
            logger.debug(f"Background spectrum:\n{background}")

    with do_step("Calculating observed (convoluted) spectrum"):
        observed = hist.Hist(rema.axes[0], storage=hist.storage.Double())
        observed.values()[:] = incident.values() @ rema.values()
        if background is not None:
            observed.values()[:] += background_scale * background.values()

    with do_step(f"Writing observed spectrum to {observed_spectrum}"):
        write_specs(
            observed_spectrum,
            {"observed": observed},
            force_overwrite,
        )


def boris2spec(
    trace_file: Path,
    output_path: Optional[Path] = None,
    var_names: Optional[List[str]] = None,
    plot: Optional[str] = None,
    plot_title: Optional[str] = None,
    plot_xlabel: Optional[str] = None,
    plot_ylabel: Optional[str] = None,
    get_mean: bool = False,
    get_median: bool = False,
    # get_mode: bool = False,
    get_variance: bool = False,
    get_std_dev: bool = False,
    get_min: bool = False,
    get_max: bool = False,
    get_hdi: bool = False,
    hdi_prob: float = one_sigma,
    force_overwrite: bool = False,
) -> None:
    """
    Creates and/or plots spectra from boris trace file.

    :param trace_file:
        Path of netcdf file containing traces generated by ``boris``.
    :param output_path:
        Path of container file which is created containing the generated
        spectra (optional).
    :param var_names: Name of variables that are evaluated.
    :param plot: Generate matplotlib window of all spectra (optional). The
        plot is displayed interactively unless a filename (different from `""`)
        is given.
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
    :param force_overwrite: Overwrite output_path if it exists.
    """
    var_names = var_names or ["incident"]
    if output_path and not force_overwrite:
        check_if_exists(output_path)

    import arviz

    trace = arviz.from_netcdf(trace_file)

    edges = trace["constant_data"]["bin_edges"]
    axis = hist.axis.Variable(edges, name="energy")

    qty_extractor = QuantityExtractor(
        mean=get_mean, median=get_median, variance=get_variance, std_dev=get_std_dev, min=get_min, max=get_max, hdi=get_hdi, hdi_prob=hdi_prob)

    res = {}
    for key, relation in trace.items():
        for var, data in relation.items():
            if var in var_names:
                if data.ndim == 1:
                    res[var] = data
                elif data.ndim == 2:
                    res.update(qty_extractor.extract(data.values, var))
                else:
                    logger.error(f"Unknown dimension {data.ndim} with shape {data.shape} for '{var}'.")

    
    specs = {
        key: (
            hist.Hist(axis, storage=hist.storage.Double(), data=val)
            if (val.ndim == 1 and val.shape[0] + 1 == edges.shape[0])
            else val
        ) for key, val in res.items()
    }

    if output_path and res:
        write_specs(output_path, specs, force_overwrite)

    if plot is not None:
        import matplotlib.pyplot as plt

        label = {
            "mean": "Mean",
            "median": "Median",
            "mode": "Mode",
            "std": "Standard Deviation",
            "var": "Variance",
            "min": "Minimum",
            "max": "Maximum",
        }

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        for var_name, props in zip(var_names, prop_cycle()):
            if var_name in res:
                plt.stairs(
                    res[var_name],
                    edges,
                    label=var_name,
                    **props,
                )
            else:
                if get_hdi:
                    plt.stairs(
                        res[f"{var_name}_hdi_hi"],
                        edges,
                        baseline=res[f"{var_name}_hdi_lo"],
                        fill=True,
                        alpha=0.3,
                        **props,
                    )

                for key in label.keys():
                    if f"{var_name}_{key}" in res:
                        plt.stairs(
                            res[f"{var_name}_{key}"],
                            edges,
                            label=f"{label[key]} ({var_name})",
                            **props,
                        )

        plt.ylim(0, None)
        plt.xlim(edges[0], edges[-1])
        if plot_title:
            plt.title(plot_title)
        if plot_xlabel:
            plt.xlabel(plot_xlabel)
        if plot_ylabel:
            plt.ylabel(plot_ylabel)
        plt.legend()
        plt.tight_layout()
        if plot == "":
            plt.show()
        else:
            plt.savefig(plot)


def make_matrix(
    dat_file_path: Path,
    output_path: Path,
    dets: Optional[List[str]] = None,
    max_energy: Optional[float] = None,
    sim_dir: Optional[Path] = None,
    force_overwrite: bool = False,
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
    :param sim_dir:
        Root of simulation directory. Paths in ``dat_file_path`` are
        given relative to this directory. If ``None``, it is assumed
        that they are given relative to ``dat_file_path``.
    :param force_overwrite: Overwrite output_path if it exists.
    """
    if not force_overwrite:
        check_if_exists(output_path)
    with do_step(f"Reading simulation dat file {dat_file_path}"):
        simulations = read_dat_file(dat_file_path, sim_dir)
        dets = dets or get_keys_in_container(simulations[0].path)

    remas = dict()
    with tqdm(dets or [None], desc="Creating matrices", unit="matrix") as pbar:
        for det in pbar:
            pbar.set_description(f"Creating matrix for detector {det}")
            sim_spectra = get_simulation_spectra(simulations, det)
            max_energy = max_energy or max(sim_spectra.keys())
            remas[det] = create_matrix(
                sim_spectra, int(max_energy), 0, max_energy
            )

    with do_step(f"Writing created matrices to {output_path}"):
        write_specs(output_path, remas, force_overwrite)


def check_matrix(
    matrix: Path,
    binning_factor: int,
    left: int,
    right: int,
    rema_name: str = "rema",
) -> None:
    """
    Visualizes detector response matrix.

    :param matrix: Path of container file containing the response matrix.
    :param binning_factor:
        Number of neighboring bins of response matrix that are merged,
        starting at ``left``.
    :param left:
        Crop response matrix to the lowest bin still containing ``left``.
    :param right:
        Crop response matrix to the highest bin not containing ``right``.
    :param rema_name:
        Name of the detector response matrix in matrix file
        (only required if not unique).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with do_step(f"Reading response matrix {rema_name} from {matrix}"):
        rema = get_rema(matrix, rema_name, binning_factor, left, right)
        logger.debug(f"Response matrix shape: {rema.shape}")
        # logger.debug(f"Response matrix diagonal:\n{rema.values().diagonal()}")

    # Hide zero-values
    rema.values()[rema.values() == 0.0] = np.nan
    vmin = np.nanquantile(rema.values()[rema.values() > 1e-20], 0.01)
    vmax = np.nanmax(rema.values())
    plt.pcolormesh(
        rema.axes[0].edges,
        rema.axes[1].edges,
        rema,
        cmap="viridis_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.title(rema_name)
    plt.xlabel("Observed energy")
    plt.ylabel("Particle energy")
    plt.ylim(rema.axes[1].edges[-1], rema.axes[1].edges[0])
    plt.tight_layout()
    plt.show()
