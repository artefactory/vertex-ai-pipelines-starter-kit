#!/usr/bin/env bash

echo "Installing virtual env..."
declare VENV_DIR=$(pwd)/venv
if ! [ -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
source venv/bin/activate

echo "Installing requirements..."
pip3 install --upgrade setuptools
pip3 install pip-tools
pip-compile requirements-ds.in
pip-compile requirements-cicd.in

pip3 install -r requirements-ds.txt
pip3 install -r requirements-cicd.txt
pip3 install ipykernel
python3 -m ipykernel install --user --name=venv
pre-commit install
echo "Installing code as a package..."
pip3 install -e .

echo "Creating Local Pipeline root folder ..."
mkdir pipelines/runs/