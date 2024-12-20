[![PyPI version](https://badge.fury.io/py/pyknos.svg)](https://badge.fury.io/py/pyknos)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyknos.svg)](https://github.com/conda-forge/pyknos-feedstock)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sbi-dev/pyknos/blob/master/CONTRIBUTING.md)
[![GitHub license](https://img.shields.io/github/license/mackelab/pyknos)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)

## Description

Python package for conditional density estimation. It either wraps or
implements diverse conditional density estimators.

### Density estimation with normalizing flows

This package provides pass-through access to all the
functionalities of [nflows](https://github.com/bayesiains/nflows).

## Installation

`pyknos` requires Python 3.8 or higher. A GPU is not required, but can lead to speed-up
in some cases. We recommend using a
[`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual environment
([Miniconda installation instructions](https://docs.conda.io/en/latest/miniconda.html)).
If `conda` is installed on the system, an environment for installing `pyknos` can be
created as follows:

```commandline
$ conda create -n pyknos_env python=3.12 && conda activate pyknos_env
```

### From PyPI

To install `pyknos` from PyPI run

```
python -m pip install pyknos
```

### From conda-forge

To install and add `pyknos` to a project with [`pixi`](https://pixi.sh/), from the project directory run

```
pixi add pyknos
```

and to install into a particular conda environment with [`conda`](https://docs.conda.io/projects/conda/), in the activated environment run

```
conda install --channel conda-forge pyknos
```

## Examples

See the [`sbi` repository](https://github.com/sbi-dev/sbi) for examples of using pyknos.

## Name

pyknós (πυκνός) is the transliterated Greek root for density
(pyknótita) and also means *sagacious*.

## Copyright notice

This program is free software: you can redistribute it and/or modify
it under the terms of the Apache License 2.0., see LICENSE for more details.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

## Acknowledgments

Thanks to Artur Bekasov, Conor Durkan and George Papamarkarios for
their work on [nflows](https://github.com/bayesiains/nflows).

The MDN implementation in this package is based on Conor M. Durkan's.
