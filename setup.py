from setuptools import find_packages, setup

setup(
    name='IPPy',
    packages=find_packages(include=['IPPy']),
    version='0.1.0',
    description='My first Python library',
    author='Davide Evangelista',
    license='MIT',
    install_requires=['numpy', 'matplotlib', 'scikit-image', 'tensorflow'],
    setup_requires=[],
)