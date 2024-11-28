# Python Project Template
This is a repository template for python projects in scientific computing. It includes all the necessary files to get started and assure the portability of the project once it is developed. The main idea is to develop a new python package 'package-name' that holds all the source code in `core`. The directory `utils` holds all the utility functions, while `data` contains the data necessary for running the projects. However, any large data sets should be stored separately from the repository.

The main scripts of the projects, where experiments are run or plots are generated, should be put in `scripts`. The directory `tests` should contain the modules to test the core code.

## Using this repository as a template
To use this repository as a template for your Python project, follow these steps:

1. Click on the "Use this template" button at the top of the repository page.
2. Fill in the necessary details for your new repository, in particular your project name `your-project-name`.
2. Clone your new repository to your local machine:

```
git clone https://github.com/your-username/your-project-name.git
```

### Developing the project
To develop a functioning project starting from this template, do the follwing:

1. Develop the core code (obviously).
2. Change the dummy values in `setup.py` to the actual values for your project.
3. Add the required packages with version information to `requirements.txt`.
4. If you actually care, write some test modules. The tests can be run by running
   ```
   pytest
   ```
   in the project directory.

### Distributing the project
To install the project on a new machine, follow these steps:

1. Clone the project and navigate to the project directory.
2. Create a new python environment, cf. [docs](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), and activate it.
3. Install the package using
    ```
    pip install .
    ```
    or using
    ```
    pip install . --editable
    ```
    if you want to edit the project.

4. Install the required packages using
   ```
   pip install -r requirements. txt
   ```


## License
This project is licensed under the [MIT License](LICENSE).