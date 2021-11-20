#!/bin/bash

# Instructions: 
# https://packaging.python.org/tutorials/packaging-projects/ 
# https://test.pypi.org/help/#apitoken

python --version

python -m pip install --upgrade build

python -m build

python3 -m pip install --upgrade twine

python3 -m twine upload --repository testpypi dist/*