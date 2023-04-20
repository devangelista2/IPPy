from setuptools import find_packages, setup

setup(
    name='IPPy',
    packages=find_packages(include=['IPPy', 'IPPy.nn', 'IPPy.tomography', 'IPPy.experimental']),
    version='1.0.0',
    description='IPPy - Inverse Problems with Python.',
    author='Davide Evangelista',
    license='MIT',
    install_requires=['numpy', 'matplotlib', 'scikit-image', 'tensorflow', 'tabulate'],
    setup_requires=[],
)
