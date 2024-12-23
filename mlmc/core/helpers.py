from typing import Callable, Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit

from mlmc.core.options import Option
from mlmc.core.estimators import coarse_fine_mc


def fit_exponential_decay(
    x: np.ndarray, y: np.ndarray, base: float = 2
) -> Tuple[float, float]:
    fun = lambda x, C, alpha: C * base ** (-alpha * x)
    popt, _ = curve_fit(fun, x, y)
    return tuple(popt)


def mlmc_pilot(
    nlevels: int,
    nsamp: int,
    h_coarse: int,
    option: Option,
    nruns: int = 10,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float = 1,
) -> Dict[str, float]:
    """
    Performs a Multilevel Monte Carlo (MLMC) pilot un to estimate the bias and variance decay.

    Parameters:
    nlevels (int): Number of levels in the MLMC hierarchy.
    nsamp (int): Number of samples to use at each level.
    h_coarse (int): Coarse level discretization parameter.
    payoff (Callable): Payoff function to evaluate.
    **payoff_kwargs: Additional keyword arguments to pass to the payoff function.

    Returns:
    Dict[str, float]: A dictionary containing:
        - 'biases': Array of bias estimates for each level.
        - 'variances': Array of variance estimates for each level.
        - 'E0': Estimated bias constant.
        - 'V0': Estimated variance constant.
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
        E0, alpha = fit_exponential_decay(np.arange(1, nlevels + 1), biases)
        V0, beta = fit_exponential_decay(np.arange(1, nlevels + 1), diff_variances)

    else:
        E0 = np.polyfit(
            np.arange(1, nlevels + 1), biases * (2 ** np.arange(1, nlevels + 1)), 0
        )[-1]
        V0 = np.polyfit(
            np.arange(1, nlevels + 1),
            diff_variances * (2.0 ** (np.arange(1, nlevels + 1))),
            0,
        )[-1]


    return {
        "biases": biases,
        "variances": diff_variances,  # now this is Var(Yfine - Ycoarse)
        "E0": E0,
        "alpha": alpha,
        "V0": V0,
        "beta": beta,
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
    optimal_nlevels = int(np.ceil(np.log2(E0 / eps) / alpha))

    if np.allclose([beta], [gamma]):

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
