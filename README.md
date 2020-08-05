[![PyPI version](https://badge.fury.io/py/pyknos.svg)](https://badge.fury.io/py/pyknos)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/sbi/blob/master/CONTRIBUTING.md)
[![GitHub license](https://img.shields.io/github/license/mackelab/pyknos)](https://github.com/mackelab/sbi/blob/master/LICENSE.txt)

## Description

Python package for conditional density estimation. It either wraps or
implements diverse conditional density estimators.

### Density estimation with normalizing flows

This package provides pass-through access to all the
functionalities of [nflows](https://github.com/bayesiains/nflows).

## Setup

Clone the repo and install all the dependencies using the
`environment.yml` file to create a conda environment: `conda env
create -f environment.yml`. If you already have a `pyknos` environment
and want to refresh dependencies, just run `conda env update -f
environment.yml --prune`.

Alternatively, you can install via `setup.py` using `pip install -e
".[dev]"` (the dev flag installs development and testing
dependencies).

## Examples

Examples are collected in notebooks in `examples/`.

## Binary files and Jupyter notebooks

### Using

We use git lfs to store large binary files. Those files are not
downloaded by cloning the repository, but you have to pull them
separately. To do so follow installation instructions here
[https://git-lfs.github.com/](https://git-lfs.github.com/). In
particular, in a freshly cloned repository on a new machine, you will
need to run both `git-lfs install` and `git-lfs pull`.

### Contributing

We use a filename filter to identify large binary files. Once you
installed and pulled git lfs you can add a file to git lfs by
appending `_gitlfs` to the basename, e.g., `oldbase_gitlfs.npy`. Then
add the file to the index, commit, and it will be tracked by git lfs.

Additionally, to avoid large diffs due to Jupyter notebook outputs we
are using `nbstripout` to remove output from notebooks before every
commit. The `nbstripout` package is downloaded automatically during
installation of `pyknos`. However, **please make sure to set up the
filter yourself**, e.g., through `nbstriout --install` or with
different options as described
[here](https://github.com/kynan/nbstripout).

## Name

pyknós (πυκνός) is the transliterated Greek root for density
(pyknótita) and also means *sagacious*.

## Copyright notice

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Acknowledgements

Thanks to Artur Bekasov, Conor Durkan and George Papamarkarios for
their work on [nflows](https://github.com/bayesiains/nflows).

The MDN implementation in this package is by Conor M. Durkan.