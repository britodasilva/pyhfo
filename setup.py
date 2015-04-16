# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:54:18 2015

@author: anderson
"""

from setuptools import setup
import version

setup(name='pyhfo',
      version=version.version,
      description='Python Package to analyse High Frequency Oscillations in ECoG data',
      url='https://github.com/andersonbrisil/pyhfo.git',
      author='Anderson Brito da Silva',
      author_email='a.brito-da-silva1@newcastle.ac.uk',
      license='MIT',
      packages=['pyhfo'],
      install_requires=['numpy','matplotlib','scipy','h5py'],
      zip_safe=False)
