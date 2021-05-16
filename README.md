# boris

[![Build Status](https://travis-ci.org/op3/boris.svg?branch=master)](https://travis-ci.org/op3/boris)
[![codecov](https://codecov.io/gh/op3/boris/branch/master/graph/badge.svg)](https://codecov.io/gh/op3/boris)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/19e37af38cc8449289b5e8abfb85f45b)](https://www.codacy.com/manual/op3/boris)
[![License: GPL-3.0](https://img.shields.io/github/license/op3/boris?color=blue)](COPYING)

## Bayesian Optimization to Reconstruct the Incident Spectrum

boris is a modern MCMC-based deconvolution code that can be used to unfold nuclear spectra.
A (simulated) detector response matrix has to be provided.
boris is programmed probabalistically using [PyMC3](https://docs.pymc.io/).
A [No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) is used to ensure fast convergence and low autocorrelation times.

## Requirements

* Python>=3.7
* [numpy](https://numpy.org/)
* [PyMC3](https://docs.pymc.io/)
* [uproot3](https://github.com/scikit-hep/uproot3) (*optional, for reading and writing root files*)
* [uproot4](https://github.com/scikit-hep/uproot4) (*optional, for reading root files*)
* [matplotlib](https://matplotlib.org/) (*optional, for plotting results*)
* [h5py](https://www.h5py.org/) (*optional, for reading and writing hdf5 files*)

## Usage

The `boris` command is provided to construct the MCMC chain:

```bash
$ boris --help
usage: boris [-h] [-l LEFT] [-r RIGHT] [-b BINNING_FACTOR] [-s SEED]
             [-c CORES] [--thin THIN] [--tune TUNE] [--burn BURN] [-n NDRAWS]
             [-H HIST] [--bg-spectrum BG_SPECTRUM] [--bg-hist BG_HIST]
             [--bg-scale BG_SCALE] [--cal-bin-centers C0 [C1 ...] |
             --cal-bin-edges C0 [C1 ...]] [--rema-name REMA_NAME]
             [--norm-hist [NORM_HIST]]
             matrixfile observed_spectrum incident_spectrum

positional arguments:
  matrixfile            container file containing detector response matrix
  observed_spectrum     txt file containing the observed spectrum
  incident_spectrum     write trace of incident spectrum to this path

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  lower edge of first bin of deconvoluted spectrum
                        (default: 0)
  -r RIGHT, --right RIGHT
                        maximum upper edge of last bin of deconvoluted
                        spectrum (default: None)
  -b BINNING_FACTOR, --binning-factor BINNING_FACTOR
                        rebinning factor, group this many bins together
                        (default: 10)
  -s SEED, --seed SEED  set random seed (default: None)
  -c CORES, --cores CORES
                        number of cores to utilize (default: 1)
  --thin THIN           thin the resulting trace by a factor (default: 1)
  --tune TUNE           number of initial steps used to tune the model
                        (default: 1000)
  --burn BURN           number of initial steps to discard (burn-in phase)
                        (default: 1000)
  -n NDRAWS, --ndraws NDRAWS
                        number of samples to draw per core (default: 2000)
  -H HIST, --hist HIST  name of histogram in observed_spectrum to read
                        (optional) (default: None)
  --bg-spectrum BG_SPECTRUM
                        path to observed background spectrum (optional)
                        (default: None)
  --bg-hist BG_HIST     name of background histogram in observed_spectrum or
                        --bg-spectrum, if specified (optional) (default: None)
  --bg-scale BG_SCALE   relative scale of background spectrum live time to
                        observed spectrum live time (optional) (default: 1.0)
  --cal-bin-centers C0 [C1 ...]
                        Provide an energy calibration for the bin centers of
                        the observed spectrum, if bins are unknown (tv style
                        calibration) (default: None)
  --cal-bin-edges C0 [C1 ...]
                        Provide an energy calibration for the bin edges of the
                        observed spectrum, if bins are unknown (default: None)
  --rema-name REMA_NAME
                        Name of the detector response matrix in matrix file
                        (default: rema)
  --norm-hist [NORM_HIST]
                        Divide detector response matrix by this histogram (e.
                        g., to correct for number of simulated particles)
                        (default: None)
```

A simple convolution of an incident spectrum using the response matrix can be performed using the `sirob` program:

```bash
$ sirob --help
usage: sirob [-h] [-l LEFT] [-r RIGHT] [-b BINNING_FACTOR] [-H HIST]
             [--bg-spectrum BG_SPECTRUM] [--bg-hist BG_HIST]
             [--bg-scale BG_SCALE] [--cal-bin-centers C0 [C1 ...] |
             --cal-bin-edges C0 [C1 ...]] [--rema-name REMA_NAME]
             [--norm-hist [NORM_HIST]]
             matrixfile incident_spectrum observed_spectrum

positional arguments:
  matrixfile            container file containing detector response matrix
  incident_spectrum     file containing the incident spectrum
  observed_spectrum     write observed (convoluted) spectrum to this path

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  lower edge of first bin of deconvoluted spectrum
                        (default: 0)
  -r RIGHT, --right RIGHT
                        maximum upper edge of last bin of deconvoluted
                        spectrum (default: None)
  -b BINNING_FACTOR, --binning-factor BINNING_FACTOR
                        rebinning factor, group this many bins together
                        (default: 10)
  -H HIST, --hist HIST  Name of histogram in incident_spectrum to read
                        (optional) (default: None)
  --bg-spectrum BG_SPECTRUM
                        path to observed background spectrum (optional)
                        (default: None)
  --bg-hist BG_HIST     name of background histogram in observed_spectrum or
                        --bg-spectrum, if specified (optional) (default: None)
  --bg-scale BG_SCALE   relative scale of background spectrum live time to
                        observed spectrum live time (optional) (default: 1.0)
  --cal-bin-centers C0 [C1 ...]
                        Provide an energy calibration for the bin centers of
                        the incident spectrum, if bins are unknown (tv style
                        calibration) (default: None)
  --cal-bin-edges C0 [C1 ...]
                        Provide an energy calibration for the bin edges of the
                        incident spectrum, if bins are unknown (default: None)
  --rema-name REMA_NAME
                        Name of the detector response matrix in matrix file
                        (default: rema)
  --norm-hist [NORM_HIST]
                        Divide detector response matrix by this histogram (e.
                        g., to correct for number of simulated particles)
                        (default: None)
```

### Input and output data

The `response_matrix` file has to contain a simulated detector response matrix.
This file can be created by the `makematrix` program:

```bash
$ makematrix --help
usage: makematrix [-h] [--sim-dir SIM_DIR] [--scale-hist-axis SCALE_HIST_AXIS]
                  [--detector [DETECTOR [DETECTOR ...]]]
                  [--max-energy [MAX_ENERGY]]
                  datfile output_path

positional arguments:
  datfile               datfile containing simulation information, each line
                        has format `<simulation_hists.root>: <energy> <number
                        of particles`
  output_path           Write resulting response matrix to this file.

optional arguments:
  -h, --help            show this help message and exit
  --sim-dir SIM_DIR     simulation file names are given relative to this
                        directory (default: Directory containing datfile)
  --scale-hist-axis SCALE_HIST_AXIS
                        Scale energy axis of histograms in case a different
                        unit is used by the simulation (default: 1.0)
  --detector [DETECTOR [DETECTOR ...]]
                        Names of histograms to create response matrices for
                        (default: All available histograms)
  --max-energy [MAX_ENERGY]
                        Maximum energy of created response matrix
```

The legacy format of [Horst](https://github.com/uga-uga/Horst)
(`.root` file with an NBINS×NBINS TH2 histogram called `rema` and a TH1 histogram called `n_simulated_particles` containing the response matrix, and the number of simulated primary particles, respectively)
is also supported.
In this case, one has to use `boris` with the arguments `--rema-name rema --norm-hist n_simulated_particles`.

The `observed_spectrum` file has to contain the experimentally observed (folded) spectrum.
It can be provided in  `.txt`, `.root`, `.hdf5` or `.txt` format (detected automatically).
If the file contains multiple objects, the name of the histogram has to be provided using the `-H`/`--hist` option.

The `incident_spectrum` file will be created by boris and contains the trace (thinned, after burn-in) of the generated MCMC chain.
Supported file formats include `.txt`, `.root`, `.hdf5` and `.npz`.
The trace will be called `incident` if supported by the file format.
Binning information is given directly (`.root`), as a separate `bin_edges` object (`.hdf5`, `.npz`) or as an additional column (`.txt`).

The `observed_spectrum` will be automatically rebinned to fit to the binning of `response_matrix`.
The `observed_spectrum` can be given with several differend binning conventions:
If it is a ROOT histogram, the assigned binning is used.
If loaded from a two-column `.txt`, `.npz` or `hdf5` array, the first column is assumed to correspond to the bin centers.
If loaded from a `.txt`, `.npz` or `hdf5` array with more than two columns, the first two columns are assumed to correspond to the lower and upper bin edges.
The resulting binning has to be contiguous.
If only one-column is given, it is assumed, that the binning corresponds to the binning of the `response_matrix`.
Using `--cal-bin-edges` or `--cal-bin-centers`, it is possible to calibrate an uncalibrated spectrum.

## Output processing

The output of boris consists of the complete generated MCMC chain.
To allow for an easy and immediate interpretation of the results,
the `boris2spec` tool is provided:

```bash
$ boris2spec --help
usage: boris2spec [-h] [--plot] [--get-mean] [--get-median] [--get-variance]
                  [--get-std-dev] [--get-min] [--get-max] [--get-hdi]
                  [--hdi-prob PROB]
                  incident_spectrum [output_path]

positional arguments:
  incident_spectrum  boris output for incident spectrum
  output_path        Write resulting spectra to this file (multiple files are
                     created for each exported spectrum if txt format is used)
                     (default: None)

optional arguments:
  -h, --help         show this help message and exit
  --plot             Display a matplotlib plot of the queried spectra
                     (default: False)
  --get-mean         Get the mean for each bin (default: False)
  --get-median       Get the median for each bin (default: False)
  --get-variance     Get the variance for each bin (default: False)
  --get-std-dev      Get the standard deviation for each bin (default: False)
  --get-min          Get the minimum for each bin (default: False)
  --get-max          Get the maximum for each bin (default: False)
  --get-hdi          Get the highest density interval for each bin (default:
                     False)
  --hdi-prob PROB    HDI prob for which interval will be computed (default:
                     0.682689492137086)
```

It can be used to export mean, median, variance, standard deviation and highest density interval (lower and upper limit).
The `incident_spectrum` argument is the output of a boris run (`.root`, `.hdf5` and `.npz` are supported).
If the `--plot` argument is provided, the chosen histograms are visualized using matplotlib.
If `output_path` is provided, the resulting histograms are written to file(s) (`.root`, `.hdf5`, `.npz` and `.txt` are supported).

## License

Copyright © 2020–2021

Oliver Papst `<opapst@ikp.tu-darmstadt.de>`

This code is distributed under the terms of the GNU General Public License, version 3 or later. See [COPYING](COPYING) for more information.

## Acknowledgements

We thank U. Friman-Gayer for valuable discussions.
This work has been funded by the State of Hesse under the grant “Nuclear Photonics” within the LOEWE program.
O. Papst acknowledges support by the Helmholtz Graduate School for Hadron and Ion Research of the Helmholtz Association.
