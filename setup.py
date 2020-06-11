# coding: utf-8

from setuptools import find_packages, setup

exec(open("pyknos/version.py").read())

setup(
    name="pyknos",
    version=__version__,
    description="conditional density estimation",
    keywords="conditional density estimation PyTorch normalizing flows mdn",
    url="https://github.com/mackelab/pyknos",
    download_url="https://github.com/mackelab/pyknos/archive/v0.11.tar.gz",
    author="Álvaro Tejero Cantero",
    author_email="alvaro@minin.es",
    packages=find_packages(exclude=["tests"]),
    license="AGPLv3",
    install_requires=[
        "matplotlib",
        "nflows==0.11",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
        "dev": [
            "autoflake",
            "black",
            "flake8",
            "isort",
            "nbstripout",
            "pytest",
            "pyyaml",
            "torchtestcase",
            "twine",
        ],
    },
    dependency_links=[],
    classifiers=[
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
