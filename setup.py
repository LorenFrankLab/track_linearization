#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "dask",
    "networkx",
]
TESTS_REQUIRE = ["pytest >= 2.7.1"]
EXTRAS_REQUIRE = {
    "numba": ["numba"],  # Optional dependency
    "ipympl": ["ipympl"],  # Optional dependency
}

setup(
    name="track_linearization",
    version="2.3.2",
    license="MIT",
    description=(""),
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
)
