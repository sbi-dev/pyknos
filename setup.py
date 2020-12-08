#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pyknos, a library for conditional density estimation.
# pyknos is licensed under the Affero General Public License v3,
# see <https://www.gnu.org/licenses/>.
#
# Note: To use the 'upload' functionality of this file, you need `twine`
#   $ pip install twine --dev

from typing import Dict, Any
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Utility functions for metadata
def path_setup_py() -> str:
    """Return absolute path of `setup.py`."""

    return os.path.abspath(os.path.dirname(__file__))


def read_long_description(fname: str, default: str) -> str:
    """Return text from file in same directory for use as the long description."""

    here = path_setup_py()
    try:
        with io.open(os.path.join(here, fname), encoding="utf-8") as f:
            return "\n" + f.read()
    except FileNotFoundError:
        return default


def load_version_dict(fname: str = "version.py") -> Dict[str, Any]:
    """Return version information in a dictionary.

    Runs the version dictionary file `name` and collects assignments there as keys in a dictionary."""

    about = dict()
    here = path_setup_py()
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, fname)) as f:
        exec(f.read(), about)

    return about


# Package meta-data.
NAME = "pyknos"
DESCRIPTION = "Conditional density estimation."
LONG_DESCRIPTION = read_long_description("README.md", default=DESCRIPTION)
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
KEYWORDS = "conditional density estimation PyTorch normalizing flows mdn"
URL = "https://github.com/mackelab/pyknos"
EMAIL = "alvaro@minin.es"
AUTHOR = "Álvaro Tejero-Cantero"
VERSION = load_version_dict()["__version__"]
PYTHON_REQUIRES = ">=3.6.0"

INSTALL_REQUIRES = (
    ["matplotlib", "nflows==0.14", "numpy", "tensorboard", "torch", "tqdm",],
)

EXTRAS_REQUIRES = (
    {
        "dev": [
            "autoflake",
            "black",
            "flake8",
            "isort",
            "nbstripout",
            "pep517",
            "pytest",
            "pytest-pep8"
            "pyyaml",
            "torchtestcase",
            "twine",
        ],
    },
)


class UploadCommand(Command):
    """Support `setup.py` upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints argument in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Clean dist directory, build source, upload, push tag to git."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(path_setup_py(), "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(VERSION))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    keywords=KEYWORDS,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license="AGPLv3",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_requires=EXTRAS_REQUIRES,
    include_package_data=True,
    classifiers=[
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # $ setup.py publish support
    cmdclass=dict(upload=UploadCommand),
)
