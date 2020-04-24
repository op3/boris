# boris

## Bayesian Optimization to Reconstruct the Incident Spectrum

boris is a modern MCMC-based deconvolution code that can be used to unfold nuclear spectra.
A (simulated) detector response matrix has to be provided.
boris is programmed probabalistically using [PyMC3](https://docs.pymc.io/).
A [No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) is used to ensure fast convergence and low autocorrelation times.

## Requirements

* Python>=3.7
* PyMC3
* uproot
* pyh5 (*optional, to write hdf5 files*)

## Usage

The `boris` command is provided to construct the MCMC chain: 

```
$ boris --help
usage: boris [-h] [-l LEFT] [-r RIGHT] [-b BIN_WIDTH] [-s SEED] [-c CORES] [--thin THIN] [--tune TUNE] [--burn BURN] [-n NDRAWS] [-H HIST]
             [--cal-bin-centers C0 [C1 ...] | --cal-bin-edges C0 [C1 ...]]
             matrix observed_spectrum incident_spectrum

positional arguments:
  matrix                response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms
  observed_spectrum     txt file containing the observed spectrum
  incident_spectrum     write trace of incident spectrum to this path

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  lower edge of first bin of deconvoluted spectrum (default: 0)
  -r RIGHT, --right RIGHT
                        maximum upper edge of last bin of deconvoluted spectrum (default: None)
  -b BIN_WIDTH, --bin-width BIN_WIDTH
                        bin width of deconvoluted spectrum (default: 10)
  -s SEED, --seed SEED  set random seed (default: None)
  -c CORES, --cores CORES
                        number of cores to utilize (default: 1)
  --thin THIN           thin the resulting trace by a factor (default: 1)
  --tune TUNE           number of initial steps used to tune the model (default: 1000)
  --burn BURN           number of initial steps to discard (burn-in phase) (default: 1000)
  -n NDRAWS, --ndraws NDRAWS
                        number of samples to draw per core (default: 2000)
  -H HIST, --hist HIST  Name of histogram in observed_spectrum to read (optional) (default: None)
  --cal-bin-centers C0 [C1 ...]
                        Provide an energy calibration for the bin centers of the observed spectrum, if bins are unknown (tv style calibration) (default: None)
  --cal-bin-edges C0 [C1 ...]
                        Provide an energy calibration for the bin edges of the observed spectrum, if bins are unknown (default: None)
```

A simple convolution of an incident spectrum using the response matrix can be performed using the `sirob` program:

```
$ sirob --help
usage: sirob [-h] [-l LEFT] [-r RIGHT] [-b BIN_WIDTH] [-H HIST] [--cal-bin-centers C0 [C1 ...] | --cal-bin-edges C0 [C1 ...]]
             matrix incident_spectrum observed_spectrum

positional arguments:
  matrix                response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms
  incident_spectrum     file containing the incident spectrum
  observed_spectrum     write observed (convoluted) spectrum to this path

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  lower edge of first bin of deconvoluted spectrum (default: 0)
  -r RIGHT, --right RIGHT
                        maximum upper edge of last bin of deconvoluted spectrum (default: None)
  -b BIN_WIDTH, --bin-width BIN_WIDTH
                        bin width of deconvoluted spectrum (default: 10)
  -H HIST, --hist HIST  Name of histogram in incident_spectrum to read (optional) (default: None)
  --cal-bin-centers C0 [C1 ...]
                        Provide an energy calibration for the bin centers of the incident spectrum, if bins are unknown (tv style calibration) (default: None)
  --cal-bin-edges C0 [C1 ...]
                        Provide an energy calibration for the bin edges of the incident spectrum, if bins are unknown (default: None)
```

#### Input and output data

The `response_matrix` file has to contain a simulated detector response matrix
(`.root` file with an NBINS×NBINS TH2 histogram called `rema` and a TH1 histogram called `n_simulated_particles` containing the response matrix, and the number of simulated primary particles, respectively).
Refer to [Horst](https://github.com/uga-uga/Horst) for more information on the required format of the response matrix.

The `observed_spectrum` file has to contain the experimentally observed (folded) spectrum.
It can be provided in  `.txt`, `.root`, `.hdf5` or `.txt` format (detected automatically).
If the file contains multiple objects, the name of the histogram has to be provided using the `-H`/`--hist` option.

The `incident_spectrum` file will be created by boris and contains the trace (thinned, after burn-in) of the generated MCMC chain.
Supported file formats include `.txt`, `.root`, `.hdf5` and `.npz`.
The trace will be called `incident` if supported by the file format.
Binning information is given directly (`.root`), as a separate `bin_edges` object (`.hdf5`, `.npz`) or as a comment (`.txt`).

Currently, it is assumed that the binning of `response_matrix` is equal to 1 keV per bin, with the lower edge of the first bin starting at 0.
The `observed_spectrum` will be automatically rebinned to fit to the binning of `response_matrix`.
The `observed_spectrum` can be given with several differend binning conventions:
If it is a ROOT histogram, the assigned binning is used.
If loaded from a two-column `.txt`, `.npz` or `hdf5` array, the first column is assumed to correspond to the bin centers.
If loaded from a `.txt`, `.npz` or `hdf5` array with more than two columns, the first two columns are assumed to correspond to the bin edges.
The resulting binning has to be contiguous.
If only one-column is given, it is assumed, that the binning corresponds to the binning of the `response_matrix`.
Using `--cal-bin-edges` or `--cal-bin-centers`, it is possible to calibrate an uncalibrated spectrum.

## License

Copyright © 2020

Oliver Papst `<opapst@ikp.tu-darmstadt.de>`

This code is distributed under the terms of the GNU General Public License. See [COPYING](COPYING) for more information.


## Acknowledgements

This work has been funded by the State of Hesse under the grant “Nuclear Photonics” within the LOEWE program.
O. Papst acknowledges support by the Helmholtz Graduate School for Hadron and Ion Research of the Helmholtz Association.
