# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:54:18 2015

@author: anderson
"""

from setuptools import setup
import version

setup(name='pyhfo',
      version=version.version,
      description='Python Package to analyse High Frequency Oscillations in electrophysiological data',
      url='https://github.com/britodasilva/pyhfo.git',
      author='Anderson Brito da Silva',
      author_email='a.brito-da-silva1@newcastle.ac.uk',
      license='MIT',
      packages=['pyhfo','pyhfo.io','pyhfo.ui','pyhfo.core'],
      install_requires=['numpy','matplotlib','scipy','h5py'],
      zip_safe=False)
