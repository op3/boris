# boris

## Bayesian Optimization to Reconstruct the Incident Spectrum

boris is a modern MCMC-based deconvolution code that can be used to unfold nuclear spectra.
A (simulated) detector response matrix has to be provided.
boris is programmed probabalistically using [PyMC3](https://docs.pymc.io/).
A [No-U-Turn Sampler](https://arxiv.org/abs/1111.4246) is used to ensure fast convergence and low autocorrelation times.

## Requirements

* Python>=3.8
* PyMC3
* uproot
* pyh5 (*optional, to write hdf5 files*)

## Usage

The `boris` command is provided to construct the MCMC chain: 

```
$ boris --help
usage: boris [-h] [-l LEFT] [-r RIGHT] [-b BIN_WIDTH] [-s SEED] [-c CORES] [--thin THIN] [--tune TUNE] [--burn BURN] [-n NDRAWS]
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
```

Refer to [Horst](https://github.com/uga-uga/Horst) for more information on the required format of the response matrix.

A simple convolution of an incident spectrum using the response matrix can be performed using the `sirob` program:

```
$ sirob --help
usage: sirob [-h] [-l LEFT] [-r RIGHT] [-b BIN_WIDTH] matrix incident_spectrum observed_spectrum

positional arguments:
  matrix                response matrix in root format, containing 'rema' and 'n_simulated_particles' histograms
  incident_spectrum     txt file containing the incident spectrum
  observed_spectrum     write observed (convoluted) spectrum to this path

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  lower edge of first bin of deconvoluted spectrum (default: 0)
  -r RIGHT, --right RIGHT
                        maximum upper edge of last bin of deconvoluted spectrum (default: None)
  -b BIN_WIDTH, --bin-width BIN_WIDTH
                        bin width of deconvoluted spectrum (default: 10)
```

## License

Copyright © 2020

Oliver Papst `<opapst@ikp.tu-darmstadt.de>`

This code is distributed under the terms of the GNU General Public License. See [COPYING](COPYING) for more information.


## Acknowledgements

This work has been funded by the State of Hesse under the grant “Nuclear Photonics” within the LOEWE program.
O. Papst acknowledges support by the Helmholtz Graduate School for Hadron and Ion Research of the Helmholtz Association.
