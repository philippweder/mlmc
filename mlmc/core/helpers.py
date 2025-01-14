from typing import Callable, Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit

from mlmc.core.options import Option
from mlmc.core.estimators import coarse_fine_mc


def fit_exponential_decay(
    x: np.ndarray, y: np.ndarray, base: float = 2
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Fit an exponential decay function to the given data.

    Parameters:
    x (np.ndarray): The independent variable data.
    y (np.ndarray): The dependent variable data.
    base (float): The base of the exponential function. Default is 2.

    Returns:
    Tuple[float, float]: The fitted parameters (C, alpha) of the exponential function.
    np.ndarray: The standard deviations of the fitted parameters.
    """
    fun = lambda x, C, alpha: C * base ** (-alpha * x)
    popt, pcov= curve_fit(fun, x, y)
    perr = np.sqrt(np.diag(pcov))
    return tuple(popt), perr


def mlmc_pilot(
    nlevels: int,
    nsamp: int,
    h_coarse: int,
    option: Option,
    nruns: int = 1,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float = 1,
) -> Dict[str, float]:
    """
    Perform a pilot run for Multilevel Monte Carlo (MLMC) to estimate parameters.

    Parameters:
    nlevels (int): Number of levels.
    nsamp (int): Number of samples.
    h_coarse (int): Coarse level step size.
    option (Option): Option object containing parameters for the simulation.
    nruns (int): Number of runs. Default is 1.
    alpha (float | None): Decay rate of the bias. Default is None.
    beta (float | None): Decay rate of the variance. Default is None.
    gamma (float): Parameter for the cost model. Default is 1.

    Returns:
    Dict[str, float]: A dictionary containing estimated parameters and statistics.
    """
    


    if (alpha is not None and alpha <= 0) or (beta is not None and beta <= 0):
        raise ValueError(f"alpha and beta must be strictly positive."
                         "Set to `None` if they should be estimated from the pilot run")
    if gamma <= 0:
        raise ValueError(f"gamma must be strictly positive. gamma = {gamma} was provided")

    levels = np.arange(0, nlevels, dtype=int)
    h_values = [h_coarse / 2**l for l in levels]

    biases = np.zeros(len(levels))
    diff_variances = np.zeros(len(levels))
    finest_variance = 0

    for run in range(nruns):

        pbar = tqdm(enumerate(h_values), desc="Running pilot run", total=len(h_values))
        for i, h in pbar:
            pbar.set_postfix({"h": h, "run": run})
            # coarse_fine_mc estimates difference between fine-level (2h) and coarse-level(h)
            result = coarse_fine_mc(nsamp, h, option)
            biases[i] += result["bias"]
            diff_variances[i] += result["var_diff"] * nsamp

            if i == nlevels - 1:
                finest_variance += result["var_coarse"] * nsamp

    biases /= nruns
    diff_variances /= nruns
    finest_variance /= nruns

    if alpha is None or beta is None:
        (E0, alpha), bias_stds = fit_exponential_decay(np.arange(1, nlevels + 1), biases)
        (V0, beta), var_stds  = fit_exponential_decay(np.arange(1, nlevels + 1), diff_variances)

    else:
        E0 = np.polyfit(
            np.arange(1, nlevels + 1), biases * (2 ** np.arange(1, nlevels + 1)), 0
        )[-1]
        V0 = np.polyfit(
            np.arange(1, nlevels + 1),
            diff_variances * (2.0 ** (np.arange(1, nlevels + 1))),
            0,
        )[-1]

        bias_stds = np.zeros(2)
        var_stds = np.zeros(2)


    return {
        "biases": biases,
        "variances": diff_variances,  # now this is Var(Yfine - Ycoarse)
        "E0": E0,
        "alpha": alpha,
        "bias_stds": bias_stds,
        "V0": V0,
        "beta": beta,
        "var_stds": var_stds,
        "Vfine": finest_variance,
    }


def compute_optimal_samps(
    E0: float,
    V0: float,
    eps: float,
    alpha: float = 1,
    beta: float = 1,
    gamma: float = 1,
) -> Tuple[int, List[int]]:
    """
    Compute the optimal number of samples for each level in MLMC.

    This function computes the optimal number of samples for each level in MLMC
    according to the complexity theorem in [1].

    [1] M. B. Giles, ‘Multilevel Monte Carlo methods’, Acta Numerica, vol. 24,
     pp. 259–328, May 2015, doi: 10.1017/S096249291500001X.


    Parameters:
    E0 (float): Initial bias estimate.
    V0 (float): Initial variance estimate.
    eps (float): Desired accuracy.
    alpha (float): Decay rate of the bias. Default is 1.
    beta (float): Decay rate of the variance. Default is 1.
    gamma (float): Parameter for the cost model. Default is 1.

    Returns:
    Tuple[int, List[int]]: The optimal number of levels and the number of samples for each level.
    """
    # optimal_nlevels = int(np.ceil(np.log2(E0 / eps) / alpha))
    optimal_nlevels = int(np.ceil((np.log2(E0 / eps) - np.log2(1 - 2 ** (-alpha))) / alpha))

    if np.allclose([beta], [gamma]): #if beta \approx gamma

        N_l = lambda l: int(np.ceil(2 ** (-l) * (optimal_nlevels + 1) * V0 / eps**2))

    elif beta > gamma:
        N_l = lambda l: int(
            np.ceil(
                V0
                * 2 ** (-(beta + gamma) * l / 2)
                / ((1 - 2 ** (-(beta - gamma) / 2)) * eps**2)
            )
        )

    else:
        N_l = lambda l: int(
            np.ceil(
                V0
                * 2 ** ((gamma - beta) * optimal_nlevels / 2)
                * 2 ** (-(beta + gamma) * l / 2)
                / ((1 - 2 ** (-(gamma - beta) / 2)) * eps**2)
            )
        )

    optimal_nsamps = [N_l(l) for l in range(optimal_nlevels + 1)]

    return optimal_nlevels, optimal_nsamps
