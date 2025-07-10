# setup.py
from setuptools import setup, find_packages

setup(
    name='delayedsw',
    version='0.1.3',
    description='Delayed space sliding window transformer',
    author='Gabriel Thaler',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'scikit-learn>=1.6.1',
    ],
    extras_require = {
        'pandas_support':  ["pandas"],
        'test' : ["pytest", "pandas"]
    }
)