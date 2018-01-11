[![Build Status](https://travis-ci.org/napaynedev/electre.svg?branch=master)](https://travis-ci.org/napaynedev/electre)
[![ReadTheDocs](https://readthedocs.org/projects/electre/badge/?version=latest)](http://electre.readthedocs.io/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/napaynedev/electre/badge.svg)](https://coveralls.io/github/napaynedev/electre)

# Overview

This repo is exploring the use of and the development of utilities to implement the ELECTRE III decision methods for software candidates.

[Documentation](http://electre.readthedocs.io/en/latest/)

# Installation

Clone the directory, then:

```
pip install -e .
```

# Usage

## Command Line Interface

```
usage: electre [-h] alternatives_csv thresholds_weights_csv

positional arguments:
  alternatives_csv      Path to CSV file containing the alternatives list with
                        scores
  thresholds_weights_csv
                        Path to CSV file containing thresholds and weights

optional arguments:
  -h, --help            show this help message and exit
```

## Data Files

The main inputs are in the CSV format.  Examples are provided in the test directory.  

# Development

It is required that you add the "pre-commit" file to your git hooks.  Move the file to the .git\hooks directory.

This will enforce flake8 style and quality.

# References

[presentation](file:///C:/Users/npayne3/Downloads/MCDA-ELECTREIII.pdf)

[PROJECT RANKING USING ELECTRE III](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.493.6585&rep=rep1&type=pdf)