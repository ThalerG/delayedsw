# setup.py
from setuptools import setup, find_packages

setup(
    name='delayedsw',
    version='0.1.0',
    description='Delayed space sliding window transformer',
    author='Gabriel Thaler',
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.0.0',
        'numpy'
    ],
)