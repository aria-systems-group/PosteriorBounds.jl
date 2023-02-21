# PosteriorBounding.jl

A toolset for bounding posterior Gaussian process (GP) regression mean and covariance components over an interval. Currently supports GPs defined with the squared-exponential kernel.

## Installation

This package is not registered (yet). Add it using `add` or `dev` in the Julia package manager, e.g.

`pkg> add https://github.com/aria-systems-group/PosteriorBounds.jl`

## Python Example

This tool can be called from Python using the `juliacall` package. Install with `pip3 juliacall`. See the example script `example/pywrapper.py` for usage details.