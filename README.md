# libeisfit

libeisfit is a shared library (libeisfit.so) and CLI application that fits equivalent circuit
models (ECMs) to measured EIS spectra. The parameters are optimised in log10 space with
Levenberg-Marquardt using an exact Jacobian from autograd through the TorchScript from libeisgenerator.
The search runs multiple starts from random points in the elements' own ranges,
and by basin hopping one parameter at a time, first inside the bounds the model string spells out and
then inside a widened box.

The library reports per-parameter values with one-sigma uncertainties, residual diagnostics
(rms, noise floor, residual autocorrelation, condition number) and a fits / marginal / does-not-fit
verdict. `eisFitBatch()` fits a whole batch of spectra against one shared frequency grid with one
backward pass serving the Jacobians of all fits.

## Compile/Install

### Requirements

* c++20 capable compiler (GCC, CLANG)
* cmake 3.23 or later
* pkg-config
* libtorch
* libeisgenerator
* libkisstype

### Procedure

* mkdir build && cd build
* cmake .. && make
* sudo make install

## Basic usage of the CLI application

Fit one circuit to one spectrum:

```
eisfit -f spec.csv -m r-rp
```

Fit every csv in a directory tree in batches and write the spectra of the fitted circuits:

```
eisfit -R -f dir/ -m 'lr-p(r-w)' -o fitted/
```

The exit code is 4 when a spectrum does not fit.

## Using the library

```cpp
#include <eisfit/fit.h>

FitResult fit = eisFit(spectra, omegas, "r-rp");
if(fit.success && fit.verdict == FIT_VERDICT_FIT)
	std::cout<<fittedModelString("r-rp", fit.parameters)<<"\n";
```

`spectra` is complex, one entry per frequency, `omegas` the angular frequencies in rad/s.
`eisFitBatch()` takes one spectrum per row against a shared grid and returns one `FitResult`
per row.
