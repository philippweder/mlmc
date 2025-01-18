# Multilevel Monte Carlo for Option Pricing
This repository contains the code generate the plots in the [report](report.pdf).


## Installation
To install the project on a new machine, follow these steps:

1. Clone the project and navigate to the project directory.
2. Create a new python environment, cf. [docs](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), and activate it.
3. Install the required packages using
   ```
   pip install -r requirements. txt
   ```
4. Install the package using
    ```
    pip install .
    ```
    or using
    ```
    pip install --editable .
    ```
    if you want to edit the project.

## Running the experiments
To run the experiments in the different sections of the report, follow these instructions. All the scripts come with an argument parser, that allows the variation of the input arguments. For help about a script `script.py`, call
```
python script.py --help
```
For the plotting scripts, add the `--usetex` flag for LaTeX support.

### Section IV.A - standard estimator
To vary the number of samples, call
```
python nsamp_variation.py
```
and then
```
python plot_nsamp_variation.py
```

To vary the time step size $h$, call
```
python h_variation.py
```
and then
```
python plot_h_variation.py
```

### Section IV.B - two-level estimator
For the two-level experiments, call
```
python two_level.py
```
and then
```
python plot_two_level.py
```
### Section IV.C - multi-level estimator
For the multi-level experiments with the Asian option, call first
```
python multi_level_asian.py
```
and then
```
python plot_multi_level_asian.py
```


### Section V. - barrier call option
For the multi-level experiments with the barrier option, call first
```
python multi_level_barrier.py
```
and then
```
python plot_multi_level_barrier.py
```

### Section VI. - variance reduction
For the experiments on variance reduction for high strike prices, run
For the multi-level experiments with the Asian option, call first
```
python higher_strike.py
```

## License
This project is licensed under the [MIT License](LICENSE).