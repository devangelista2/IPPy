from setuptools import setup, find_packages

setup(
    name="inverse-problems-Python",
    version="1.0.1",
    description="Inverse Problems with Python.",
    author="Davide Evangelista",
    author_email="davide.evangelista5@unibo.it",
    url="https://github.com/devangelista2/IPPy",
    packages=find_packages(),  # Automatically finds all packages in your library
    install_requires=[  # List any dependencies here
        "numpy",
        "torch",
        "matplotlib",
        "torchvision",
        "numba",
        "astra-toolbox",
        "scikit-image",
        "Pillow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or whatever license you're using
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Specify the required Python version
)
