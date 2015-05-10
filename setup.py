# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:54:18 2015

@author: anderson
"""

from setuptools import setup
import io
import version


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.rst')
setup(name='pyhfo',
      version=version.version,
      description='Python Package to analyse High Frequency Oscillations in electrophysiological data',
      long_description=long_description,
      url='https://github.com/britodasilva/pyhfo.git',
      download_url = 'https://github.com/britodasilva/pyhfo/tarball/1.1.0',
      author='Anderson Brito da Silva',
      author_email='a.brito-da-silva1@newcastle.ac.uk',
      license='MIT',
      packages=['pyhfo','pyhfo.io','pyhfo.ui','pyhfo.core'],
      install_requires=['numpy','matplotlib','scipy','h5py','sklearn'],
      zip_safe=False)
