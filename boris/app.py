#!/usr/bin/env python
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

"""boris command line interface"""

import sys
import argparse
import contextlib
from pathlib import Path
from typing import Union, List, Generator

import numpy as np

if __name__ == "__main__":
    project_dir = Path(__file__).absolute().parents[1].resolve()
    project_path = str((project_dir/"boris").resolve())
    if project_path in sys.path:
        sys.path.remove(project_path)
    sys.path.insert(0, str(project_dir))

from boris import rebin_hist, get_rema, deconvolute

@contextlib.contextmanager
def do_step(text: str) -> Generator[None, None, None]:
    print(f"{text} ...")
    try:
        yield
        print(f"{text} complete")
    except BaseException as e:
        print(f"{text} failed:")
        print(e)
        exit(1)

def write_hist(
        histfile: Path,
        name: str,
        hist: np.array,
        bin_edges: np.array) -> None:
    if histfile.exists():
        raise Exception(f"Error writing {histfile}, already exists.")
    if histfile.suffix == ".npz":
        np.savez_compressed(
            histfile,
            **{
                name: hist,
                "bin_edges": bin_edges,
            })
    elif histfile.suffix == ".txt":
        np.savetxt(
            histfile,
            hist,
            header="bin edges:\n" + str(bin_edges) + f"\n{name}:")
    elif histfile.suffix == ".hdf5":
        try:
            import h5py
            with h5py.File(histfile, "w") as f:
                f.create_dataset(
                    name, data=hist,
                    compression="gzip", compression_opts=9)
                f.create_dataset(
                    "bin_edges", data=bin_edges,
                    compression="gzip", compression_opts=9)
        except ModuleNotFoundError:
            raise Exception("Please install h5py to write hdf5 files")
    elif histfile.suffix == ".root":
        import uproot
        # TODO: Discard sumw2?
        if hist.ndim == 1:
            from uproot_methods.classes.TH1 import from_numpy
            h = from_numpy([hist, bin_edges])
        else:
            from uproot_methods.classes.TH2 import from_numpy
            h = from_numpy([hist, np.arange(0, hist.shape[0] + 1), bin_edges])
        with uproot.create(histfile) as f:
            f[name] = h
    else:
        raise Exception(f"Unknown output format: {histfile.suffix}")

def read_spectrum(spectrum: Path, histname: Union[None, str] = None) -> np.ndarray:
    """
    Read spectrum to numpy array
    """
    if spectrum.suffix == ".root":
        import uproot
        with uproot.open(spectrum) as specfile:
            if histname:
                return specfile[histname].values
            elif len(list(specfile.keys())) == 1:
                return specfile[list(specfile.keys())[0]].values
            else:
                raise Exception("Please provide name of histogram to read from root file.")
    elif spectrum.suffix == ".npz":
        with np.load(spectrum) as specfile:
            if histname:
                return specfile[histname]
            elif len(list(specfile.keys())) == 1:
                return specfile[list(specfile.keys())[0]]
            else:
                raise Exception("Please provide name of histogram to read from npz file.")
    elif spectrum.suffix == ".hdf5":
        import h5py
        with h5py.File(spectrum) as specfile:
            if histname:
                return specfile[histname]
            elif len(list(specfile.keys())) == 1:
                return specfile[list(specfile.keys())[0]]
            else:
                raise Exception("Please provide name of histogram to read from hdf5 file.")
    else:
        return np.loadtxt(spectrum)

def boris(
        matrix: Path,
        observed_spectrum: Path,
        incident_spectrum: Path,
        bin_width: int,
        left: int,
        right: int,
        ndraws: int,
        tune: int,
        thin: int,
        burn: int,
        cores: int,
        histname: Union[None, str] = None) -> None:
    """
    Load response matrix and spectrum, sample MCMC chain,
    write resulting trace to file.

    Args:
        matrix: Path of response matrix in root format
        observed_spectrum: Read observed spetcrum from this path
        incident_spectrum: Write incident spectrum trace to this path
        histname: name of histogram in observed_spectrum to read (optional)
    """
    with do_step(f"Reading response matrix {matrix}"):
        rema_counts, rema_nsim, rema_bin_edges = get_rema(
            matrix, bin_width, left, right)
        rema = rema_counts / rema_nsim

    with do_step(f"Reading observed spectrum {observed_spectrum}"):
        read_spectrum(observed_spectrum, histname)
        spectrum, spectrum_bin_edges = rebin_hist(
            spectrum, bin_width, left, right)

    with do_step(f"Sampling from posterior distribution"):
        trace = deconvolute(
            rema,
            spectrum,
            ndraws=ndraws,
            tune=tune,
            thin=thin,
            burn=burn,
            cores=cores)

    with do_step(f"Writing incident spectrum trace to {incident_spectrum}"):
        write_hist(incident_spectrum, "incident",
                    trace["incident"], rema_bin_edges)

class BorisApp:
    def __init__(self) -> None:
        args = self.parse_args(sys.argv[1:])
        if args.seed:
            with do_step(f"Setting numpy seed to {args.seed}"):
                np.random.seed(int(args.seed))
        boris(args.matrix, args.observed_spectrum, args.incident_spectrum,
              args.bin_width, args.left, args.right, args.ndraws,
              args.tune, args.thin, args.burn, args.cores, args.hist)

    def parse_args(self, args: List[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "-l", "--left",
            help="lower edge of first bin of deconvoluted spectrum",
            type=int,
            default=0
        )
        parser.add_argument(
            "-r", "--right",
            help="maximum upper edge of last bin of deconvoluted spectrum",
            type=int,
            default=None
        )
        parser.add_argument(
            "-b", "--bin-width",
            help="bin width of deconvoluted spectrum",
            type=int,
            default=10
        )
        parser.add_argument(
            "-s", "--seed",
            help="set random seed"
        )
        parser.add_argument(
            "-c", "--cores",
            help="number of cores to utilize",
            default=1,
            type=int
        )
        parser.add_argument(
            "--thin",
            help="thin the resulting trace by a factor",
            default=1,
            type=int
        )
        parser.add_argument(
            "--tune",
            help="number of initial steps used to tune the model",
            default=1000,
            type=int
        )
        parser.add_argument(
            "--burn",
            help="number of initial steps to discard (burn-in phase)",
            default=1000,
            type=int
        )
        parser.add_argument(
            "-n", "--ndraws",
            help="number of samples to draw per core",
            default=2000,
            type=int
        )
        parser.add_argument(
            "-H", "--hist",
            help="Name of histogram in observed_spectrum to read (optional)",
            default=None,
            type=str
        )
        parser.add_argument(
            "matrix",
            help="response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms",
            type=Path
        )
        parser.add_argument(
            "observed_spectrum",
            help="txt file containing the observed spectrum",
            type=Path,
        )
        parser.add_argument(
            "incident_spectrum",
            help="write trace of incident spectrum to this path",
            type=Path,
        )
        
        return parser.parse_args(args)

def sirob(
        matrix: Path,
        incident_spectrum: Path,
        observed_spectrum: Path,
        bin_width: int,
        left: int,
        right: int,
        histname: Union[None, str] = None) -> None:
    with do_step(f"Reading response matrix from {matrix}"):
        rema_counts, rema_nsim, rema_bin_edges = get_rema(
            matrix, bin_width, left, right)
        rema = rema_counts / rema_nsim

    with do_step(f"Reading incident spectrum from {incident_spectrum}"):
        spectrum = read_spectrum(incident_spectrum, histname)
        incident, spectrum_bin_edges = rebin_hist(
            spectrum, bin_width, left, right)
    
    with do_step("Calculating observed (convoluted) spectrum"):
        observed = incident @ rema

    with do_step(f"Writing observed spectrum to {observed_spectrum}"):
        write_hist(observed_spectrum, "observed",
                   observed, spectrum_bin_edges)
    
        
class SirobApp:
    def __init__(self) -> None:
        args = self.parse_args(sys.argv[1:])
        sirob(args.matrix, args.incident_spectrum, args.observed_spectrum,
              args.bin_width, args.left, args.right, args.hist)

    def parse_args(self, args: List[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "-l", "--left",
            help="lower edge of first bin of deconvoluted spectrum",
            type=int,
            default=0
        )
        parser.add_argument(
            "-r", "--right",
            help="maximum upper edge of last bin of deconvoluted spectrum",
            type=int,
            default=None
        )
        parser.add_argument(
            "-b", "--bin-width",
            help="bin width of deconvoluted spectrum",
            type=int,
            default=10
        )
        parser.add_argument(
            "-H", "--hist",
            help="Name of histogram in incident_spectrum to read (optional)",
            default=None,
            type=str
        )
        parser.add_argument(
            "matrix",
            help="response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms",
            type=Path
        )
        parser.add_argument(
            "incident_spectrum",
            help="file containing the incident spectrum",
            type=Path
        )
        parser.add_argument(
            "observed_spectrum",
            help="write observed (convoluted) spectrum to this path",
            type=Path
        )
        return parser.parse_args(args)
        

if __name__ == "__main__":
    if Path(sys.argv[0]).stem == "sirob":
        SirobApp()
    else:
        BorisApp()