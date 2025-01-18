from setuptools import setup, find_packages

setup(
    name='mlmc',       # Replace with your package name
    version='0.1',                  # Package version
    author='Philipp Weder',             # Author name
    author_email='wederphil@gmail.com',  # Author email
    description='Implementation of the multi-level Monte Carlo method for option pricing',  # Short description
    long_description=
    """
    # MLMC: Multi-Level Monte Carlo Method for Option Pricing

    This package provides an implementation of the multi-level Monte Carlo (MLMC) method for option pricing. The MLMC method is a powerful technique used in computational finance to efficiently estimate the expected value of a financial option by combining simulations at different levels of accuracy.

    ## Features

    - Efficient option pricing using the MLMC method
    - Support for various types of options
    - Easy-to-use API for integrating with other financial models

    ## Installation

    You can install the package using pip:
    ```
    pip install -e .
    ```
    """,
    long_description_content_type='text/markdown',  # Description content type
    url='https://github.com/philippweder/mlmc',  # URL to your package repository
    packages=find_packages(),       # Automatically find all packages in the directory
    install_requires=[              # List of dependencies required to run your package
        # Add more dependencies as needed
    ],
    classifiers=[                   # Classifiers to categorize your package (optional)
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        # Add more classifiers as needed
    ],
)
