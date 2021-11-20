#!/bin/bash

# Instructions: 
# https://packaging.python.org/tutorials/packaging-projects/ 
# https://test.pypi.org/help/#apitoken

python --version

python -m pip install --upgrade build

python -m build

python3 -m pip install --upgrade twine

name=`python read_config.py --key "name" | tail -1`
version=`python read_config.py --key "version" | tail -1`

python3 -m twine upload --repository testpypi dist/${name}-${version}*