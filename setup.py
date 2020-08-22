#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib',
                    'pandas', 'dask', 'networkx']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='track_linearization',
    version='1.0.0',
    license='MIT',
    description=(''),
    author='',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
