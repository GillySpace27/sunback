Solar Background Updater
==============================
[//]: # (Badges)
<a href="https://github.com/GillySpace27/sunback/tree/master" target="_blank">![GitHub last commit (branch)](https://img.shields.io/github/last-commit/GillySpace27/sunback/master)</a>
<a href="https://travis-ci.com/GillySpace27/sunback" target="_blank">![Travis Build Status](https://travis-ci.com/GillySpace27/sunback.svg?branch=master)</a>
<a href="https://ci.appveyor.com/project/GillySpace27/sunback/" target="_blank">![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/ji7e0pm5xxckf6rq/branch/master?svg=true)</a>
<a href="https://sunback.readthedocs.io/en/latest/?badge=latest" target="_blank">![Documentation Status](https://readthedocs.org/projects/sunback/badge/?version=latest)</a>

<a href="https://pypi.org/project/sunback/" target="_blank"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/sunback"></a>
<a href="https://pypi.org/project/sunback/" target="_blank">![PyPI](https://img.shields.io/pypi/v/sunback)</a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/sunback">
<a href="https://github.com/GillySpace27/sunback/" target="_blank">![GitHub repo size](https://img.shields.io/github/repo-size/GillySpace27/sunback)</a>
<a href="https://opensource.org/licenses/BSD-3-Clause" target="_blank">![GitHub](https://img.shields.io/github/license/GillySpace27/sunback)</a>

<a href="https://codeclimate.com/github/GillySpace27/sunback/maintainability"><img src="https://api.codeclimate.com/v1/badges/f7ae86dc9703d4a7eec6/maintainability" /></a>
<a href="https://codeclimate.com/github/GillySpace27/sunback/test_coverage"><img src="https://api.codeclimate.com/v1/badges/f7ae86dc9703d4a7eec6/test_coverage" /></a>
<a href="https://www.codacy.com/manual/GillySpace27/sunback" target="_blank">![Codacy Badge](https://api.codacy.com/project/badge/Grade/a47b3701e7544010a4708d923a71fedb)</a>
<a href="https://codecov.io/gh/GillySpace27/sunback/branch/master" target="_blank">![codecov](https://codecov.io/gh/GillySpace27/sunback/branch/master/graph/badge.svg)</a>


This program is the core analysis package that Dr. Gilly uses for his research. This code is very modular and can perform many different data pipelines for images with arbitrary and custom filtering steps. Sunback runs analysis on images from AIA, updates Gilly's personal website, and allows users to run a daemon on their machines to update their computer desktop background.

## Installation

### Github Installation Instructions

``` zsh
git clone https://github.com/GillySpace27/sunback.git
cd sunback
python3 -m venv .venv
source .venv/bin/activate
pip install .
sunback-run
```

### PyPi Installation Instructions:
``` bash
pip install sunback
```
followed by either
```
sunback-run
```
in the console, or in python
```
python3
>> from sunback.run.run_client_background import run_client as run
>> run(delay=20) # Delay in seconds between updates
```

Check contribute.md to see how you can help out!

### Copyright

Copyright (c) 2019, [C. R. Gilly](https://gilly.space) [ORCiD](https://orcid.org/0000-0003-0021-9056)


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
