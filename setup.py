"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='laminar_uecog_viz',
      description='Laminear and uECoG plotting scripts.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      version='0.0.1',
      author='',
      author_email='',
      packages=find_packages())
      #scripts=['scripts/preprocess_folder'])
