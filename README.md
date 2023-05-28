# Molecule-Gym

Molecule-Gym is a library to run machine learning experiments on drug discovery datasets.

## Contents

- [Installation](#installation)

## Installation

- Initializa the repository to generate all the auxiliary data and results storing with
```bash
./init.sh
```
- Create virtual environment
```bash
mkdir -p ~/venvs/molecule_gym
python3 -m venv ~/venvs/molecule_gym
```
- Activate virtual environment
```bash
source ~/venvs/molecule_gym/bin/activate
```
- Upgrade pip and setuptools
```bash
pip install --upgrade pip setuptools
```
- Change to the repo directory and install dependencies with
```bash
pip install -r requirements.txt
```
- Install the library in development mode with
```bash
python setup.py develop
```
- You can check the packages installed in the virtual environment with
```bash
pip freeze
```
