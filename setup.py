#!/usr/bin/env python3
# flake8: noqa
'''
Setup script for the repository.
'''
from setuptools import setup, find_packages, Distribution
from molecule_gym.version import __version__

def read_requirements():
    """
    TODO: currently does not read URLs correctly.
    """
    with open('requirements.txt') as req:
        requirements = req.read().split('\n')
    return requirements

setup(name='molecule-gym',
      version=__version__,
      description='Machine learning library for in-silico drug discovery experiments',
      distclass=Distribution,
      install_requires=[], #read_requirements(),
      packages=find_packages(include=['molecule_gym', 'molecule_gym.*']),
      author='Pranjal Dhole',
      author_email='dhole.pranjal@gmail.com',
      zip_safe=False,
)