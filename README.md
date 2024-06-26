# boris

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11147261.svg)](https://doi.org/10.5281/zenodo.11147261)
[![Test Status](https://img.shields.io/github/actions/workflow/status/op3/boris/run-tests.yml?branch=master&label=tests)](https://github.com/op3/boris/actions/workflows/run-tests.yml)
[![Code Style](https://img.shields.io/github/actions/workflow/status/op3/boris/check-code-formatting.yml?branch=master&label=style)](https://github.com/op3/boris/actions/workflows/check-code-formatting.yml)
[![codecov](https://codecov.io/gh/op3/boris/branch/master/graph/badge.svg)](https://codecov.io/gh/op3/boris)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/19e37af38cc8449289b5e8abfb85f45b)](https://www.codacy.com/manual/op3/boris)
[![License: GPL-3.0](https://img.shields.io/github/license/op3/boris?color=blue)](COPYING)

## Bayesian Operations to Reconstruct the Incident Spectrum

boris is a modern MCMC-based code that can be used
to reconstruct incident nuclear spectra via forward fitting.
A (simulated) detector response matrix has to be provided.
boris is programmed probabalistically using [PyMC](https://docs.pymc.io/).
A [No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) is used to ensure fast convergence and low autocorrelation times.

## Features

- Detector response matrix generation (`makematrix`)
    - From individual spectra for different energies (e.g. from [Geant4](https://geant4.web.cern.ch/) simulations)
    - Linear interpolation for missing energies
    - Visualize detector response matrix to check correctness (`checkmatrix`)
- Reconstruction of incident spectra from observed spectra using detector response matrix (`boris`)
- Advanced treatment of uncertainties
- Subtraction of background spectra with correct treatment of uncertainties
- Supports reading and writing txt, numpy, hdf5 and root formats
- Rebin and calibrate input spectra
- Supply two detector response matrices and find the best interpolation between the two
    - Useful if additional unknown degree of freedom, such as detector offset, is present
- Directly fit gaussian beam profile model (with left tail) to extract beam properties
- Extract (and plot) statistical summary spectra such as mean or standard deviation from MCMC-traces (`boris2spec`)
- Convolute spectrum with detector response matrix (`sirob`)

## Requirements

* Python>=3.10
* [numpy](https://numpy.org/)
* [PyMC](https://docs.pymc.io/)>=5.1
* [arviz](https://python.arviz.org/)>=0.14
* [NumPyro](https://num.pyro.ai/) (*optional, for NumPyro backend, faster execution times!*)
* [uproot](https://github.com/scikit-hep/uproot4)>=4.1 (*optional, for reading and writing root files*)
* [h5py](https://www.h5py.org/) (*optional, for reading and writing hdf5 files*)
* [matplotlib](https://matplotlib.org/) (*optional, for plotting results*)

## Installation

You can install boris using pip or pipx, e.g.:

```
pip install git+https://github.com/op3/boris.git
```

## Usage

The `boris` command is provided to construct the MCMC chain:

```bash
$ boris --help
usage: boris [-h] [-v] [-l LEFT] [-r RIGHT] [-b BINNING_FACTOR] [-H HIST]
             [--bg-spectrum BG_SPECTRUM] [--bg-hist BG_HIST]
             [--bg-scale BG_SCALE] [--rema-name [REMA_NAME]]
             [--matrixfile-alt [MATRIXFILE_ALT]]
             [--cal-bin-centers C0 [C1 ...] | --cal-bin-edges C0 [C1 ...]]
             [-s SEED] [-c CORES] [--tune TUNE] [-n NDRAWS] [--fit-beam]
             [--force-overwrite]
             matrixfile observed_spectrum incident_spectrum

Reconstruct incident_spectrum based on observed_spectrum using the supplied detector response matrix.

positional arguments:
  matrixfile            container file containing detector response matrix
  observed_spectrum     file containing the observed spectrum
  incident_spectrum     write trace of incident spectrum to this path (.h5
                        file)

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity
  -l LEFT, --left LEFT  crop spectrum to the lowest bin still containing LEFT
                        (default: 0)
  -r RIGHT, --right RIGHT
                        crop spectrum to the highest bin not containing RIGHT
                        (default: maximum energy of simulation)
  -b BINNING_FACTOR, --binning-factor BINNING_FACTOR
                        rebinning factor, group this many bins together
                        (default: 10)
  -H HIST, --hist HIST  name of histogram in observed_spectrum to read
                        (optional) (default: None)
  --bg-spectrum BG_SPECTRUM
                        path to observed background spectrum (optional)
                        (default: None)
  --bg-hist BG_HIST     name of background histogram in observed_spectrum or
                        --bg-spectrum, if specified (optional) (default: None)
  --bg-scale BG_SCALE   relative scale of background spectrum live time to
                        observed spectrum live time (optional) (default: 1.0)
  --rema-name [REMA_NAME]
                        name of the detector response matrix in matrix file
                        (default: rema)
  --matrixfile-alt [MATRIXFILE_ALT]
                        Load an additional detector response matrix from this
                        matrix file (same rema-name as main matrix). Boris
                        will create a linear combination of the main
                        matrixfile and the alternative matrix file. (default:
                        None)
  --cal-bin-centers C0 [C1 ...]
                        energy calibration for the bin centers of the observed
                        spectrum, if bins are unknown (tv style calibration)
                        (default: None)
  --cal-bin-edges C0 [C1 ...]
                        energy calibration for the bin edges of the observed
                        spectrum, if bins are unknown (default: None)
  --force-overwrite     Overwrite existing files without warning

advanced arguments:
  -s SEED, --seed SEED  set random seed
  -c CORES, --cores CORES
                        number of cores to utilize (default: 1)
  --tune TUNE           number of initial steps used to tune the model
                        (default: 1000)
  -n NDRAWS, --ndraws NDRAWS
                        number of samples to draw per core (default: 2000)
  --fit-beam            Perform a fit of a beam profile (default: False)
```

A simple convolution of an incident spectrum using the response matrix can be performed using the `sirob` program:

```bash
$ sirob --help
usage: sirob [-h] [-v] [-l LEFT] [-r RIGHT] [-b BINNING_FACTOR] [-H HIST]
             [--bg-spectrum BG_SPECTRUM] [--bg-hist BG_HIST]
             [--bg-scale BG_SCALE] [--cal-bin-centers C0 [C1 ...] |
             --cal-bin-edges C0 [C1 ...]] [--rema-name [REMA_NAME]]
             [--force-overwrite]
             matrixfile incident_spectrum observed_spectrum

positional arguments:
  matrixfile            container file containing detector response matrix
  incident_spectrum     file containing the incident spectrum
  observed_spectrum     write observed (convoluted) spectrum to this path

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity (default: False)
  -l LEFT, --left LEFT  crop spectrum to the lowest bin still containing LEFT
                        (default: 0)
  -r RIGHT, --right RIGHT
                        crop spectrum to the highest bin not containing RIGHT
                        (default: maximum energy of simulation) (default:
                        None)
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
  --rema-name [REMA_NAME]
                        Name of the detector response matrix in matrix file
                        (default: rema)
  --force-overwrite     Overwrite existing files without warning (default:
                        False)
```

### Input and output data

The `response_matrix` file has to contain a simulated detector response matrix.
This file can be created by the `makematrix` program:

```bash
$ makematrix --help
usage: makematrix [-h] [-v] [--sim-dir SIM_DIR] [--detector [DETECTOR ...]]
                  [--max-energy [MAX_ENERGY]] [--force-overwrite]
                  datfile output_path

Create a detector response matrix from multiple simulated spectra for
different energies.

positional arguments:
  datfile               datfile containing simulation information, each line
                        has format `<simulation_hists.root>: <energy> [<number
                        of particles>]`
  output_path           Write resulting response matrix to this file.

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity
  --sim-dir SIM_DIR     simulation file names are given relative to this
                        directory (default: Directory containing datfile)
  --detector [DETECTOR ...]
                        Names of histograms to create response matrices for
                        (default: All available histograms)
  --max-energy [MAX_ENERGY]
                        Maximum energy of created response matrix
  --force-overwrite     Overwrite existing files without warning
```

The provided datfile contains a list of all simulated spectra
that should be combined into a single detector response matrix.
For each incident particle energy,
you have to provide one spectrum in separate files.
If your spectra are already normalized to the number of simulated particles,
the datfile should look like this:

```
sim_1000keV.root: 1000
sim_2000keV.root: 2000
sim_3000keV.root: 3000
sim_4000keV.root: 4000
```

If your spectra are not normalized (i.e., divided by the number of simulated particles),
you have to provide the number of simulated particles as well:

```
sim_1000keV.root: 1000 1000000000
```

If the files contain spectra for multiple detectors,
detector response matrices are created for all of them,
unless you restrict them using the `--detector` argument.

The legacy format of [Horst](https://github.com/uga-uga/Horst)
(`.root` file with an NBINS×NBINS TH2 histogram called `rema` and a TH1 histogram called `n_simulated_particles` containing the response matrix, and the number of simulated primary particles, respectively)
is also supported.
In this case, one has to use `boris` with the arguments `--rema-name rema --norm-hist n_simulated_particles`.

The `observed_spectrum` file has to contain the experimentally observed spectrum.
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
usage: boris2spec [-h] [-v] [--var-names [VAR_NAMES ...]] [--plot [OUTPUT]]
                  [--plot-title PLOT_TITLE] [--plot-xlabel PLOT_XLABEL]
                  [--plot-ylabel PLOT_YLABEL] [--get-mean] [--get-median]
                  [--get-variance] [--get-std-dev] [--get-min] [--get-max]
                  [--get-hdi] [--hdi-prob PROB] [--burn NUMBER]
                  [--thin FACTOR] [--force-overwrite]
                  trace_file [output_path]

Create spectra from boris trace files

positional arguments:
  trace_file            boris output containing traces
  output_path           Write resulting spectra to this file (multiple files
                        are created for each exported spectrum if txt format
                        is used) (default: None)

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity (default: False)
  --var-names [VAR_NAMES ...]
                        Names of variables that are evaluated (default:
                        ['spectrum', 'incident_scaled_to_fep'])
  --plot [OUTPUT]       Generate a matplotlib plot of the queried spectra. The
                        plot is displayed interactively unless an output
                        filename is given. (default: None)
  --plot-title PLOT_TITLE
                        Set plot title (default: None)
  --plot-xlabel PLOT_XLABEL
                        Set plot x-axis label (default: None)
  --plot-ylabel PLOT_YLABEL
                        Set plot y-axis label (default: None)
  --get-mean            Get the mean for each bin (default: False)
  --get-median          Get the median for each bin (default: False)
  --get-variance        Get the variance for each bin (default: False)
  --get-std-dev         Get the standard deviation for each bin (default:
                        False)
  --get-min             Get the minimum for each bin (default: False)
  --get-max             Get the maximum for each bin (default: False)
  --get-hdi             Get the highest density interval for each bin
                        (default: False)
  --hdi-prob PROB       HDI prob for which interval will be computed (default:
                        0.682689492137086)
  --burn NUMBER         Skip initial NUMBER of samples to account for burn-in
                        phase during sampling (default: 1000)
  --thin FACTOR         Thin trace by FACTOR before evaluating to reduce
                        autocorrelation (default: 1)
  --force-overwrite     Overwrite existing files without warning (default:
                        False)
```

It can be used to export mean, median, variance, standard deviation and highest density interval (lower and upper limit).
The `incident_spectrum` argument is the output of a boris run (`.root`, `.hdf5` and `.npz` are supported).
If the `--plot` argument is provided, the chosen histograms are visualized using matplotlib.
If `output_path` is provided, the resulting histograms are written to file(s) (`.root`, `.hdf5`, `.npz` and `.txt` are supported).

Furthermore, the `checkmatrix` tool is available to view detector response matrices:

```bash
$ checkmatrix --help
usage: checkmatrix [-h] [-v] [-l LEFT] [-r RIGHT] [-b BINNING_FACTOR]
                   [--rema-name [REMA_NAME]]
                   matrixfile

Display detector response matrix.

positional arguments:
  matrixfile            container file containing detector response matrix

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity
  -l LEFT, --left LEFT  crop response matrix to the lowest bin still
                        containing LEFT (default: 0)
  -r RIGHT, --right RIGHT
                        crop response matrix to the highest bin not containing
                        RIGHT (default: maximum energy of simulation)
  -b BINNING_FACTOR, --binning-factor BINNING_FACTOR
                        rebinning factor, group this many bins together
                        (default: 10)
  --rema-name [REMA_NAME]
                        name of the detector response matrix in matrix file
                        (default: rema)
```

### Beam profile model

During the reconstruction, a fit of a beam profile can be applied.
The following function is used for this purpose,
which corresponds to a gaussian distribution with a left exponential tail:

```python
def beam_profile_model(x, pos, vol, sigma, tl):
    tl = 1. / (tl * sigma)
    dx = x - pos
    norm = 1 / (
        (sigma ** 2) / tl * np.exp(-(tl * tl) / (2.0 * sigma ** 2))
        + np.sqrt(np.pi / 2.0) * sigma * (1 + np.math.erf(tl / (np.sqrt(2.0) * sigma)))
    )
    _x = np.piecewise(
        dx,
        [dx < -tl],
        [
            lambda dx: tl / (sigma ** 2) * (dx + tl / 2.0),
            lambda dx: -dx * dx / (2.0 * sigma ** 2),
        ],
    )
    return vol * norm * np.exp(_x)
```

## License

Copyright ©

Oliver Papst `<opapst@ikp.tu-darmstadt.de>`

This code is distributed under the terms of the GNU General Public License, version 3 or later. See [COPYING](COPYING) for more information.

## Acknowledgements

We thank U. Friman-Gayer for valuable discussions
and J. Kleemann for testing.
This work has been funded by the State of Hesse under the grant “Nuclear Photonics” within the LOEWE program (LOEWE/2/11/519/03/04.001(0008)/62),
and by the Bundesministerium für Bildung und Forschung (BMBF, Federal Ministry of Education and Research) under grant 05P21RDEN9.
O. Papst acknowledges support by the Helmholtz Graduate School for Hadron and Ion Research of the Helmholtz Association.
