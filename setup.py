from setuptools import setup, find_packages

setup(
    name='mlmc',       # Replace with your package name
    version='0.1',                  # Package version
    author='Your Name',             # Author name
    author_email='your@email.com',  # Author email
    description='Description of your package',  # Short description
    long_description='Long description of your package',  # Full description
    long_description_content_type='text/markdown',  # Description content type
    url='https://github.com/your_username/your_package_name',  # URL to your package repository
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
