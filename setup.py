#!/usr/bin/env python

import align
from setuptools import setup

setup(name='align',
      version=align.__version__,
      description='Simple alignment and registration of 1-dimensional signals.',
      long_description=open('README.md').read(),
      author='Remy Prechelt',
      author_email='prechelt@hawaii.edu',
      url='https://github.com/rprechelt/align',
      install_requires=['numpy', 'scipy'],
      packages=['align'],
      license='GPLv3',
      keywords=['signal', 'alignment', 'processing'],
      python_requires='>=3.5',
      tests_require=['pytest>3']
      )
