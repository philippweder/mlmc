import numpy as np
from typing import Dict, Callable, List
from numba import jit


# speed up computations using numba
@jit(nopython=True)
def simulate_path(
    S: np.ndarray, r: float, h: float, sigma: float, dW: np.ndarray, nsteps: int
) -> np.ndarray:

    sqrt_h = np.sqrt(h)

    for m in range(nsteps):
        # Update stock price using Euler-Maruyama scheme
        S[m + 1] = (1.0 + r * h + sigma * sqrt_h * dW[m]) * S[m]
    return S


@jit(nopython=True)
def batch_simulate_path(
    S: np.ndarray,
    r: float,
    h: float,
    sigma: float,
    dW: np.ndarray,
    nsteps: int,
    nsamp: int,
) -> np.ndarray:

    nsamp = S.shape[0]
    sqrt_h = np.sqrt(h)

    for n in range(nsamp):
        for m in range(nsteps):
            # Update stock price using Euler-Maruyama scheme
            S[n, m + 1] = (1.0 + r * h + sigma * sqrt_h * dW[n, m]) * S[n, m]

    return S


def standard_mc(
    nsamp: int, h: float, payoff: Callable, **payoff_kwargs
) -> Dict[str, float]:
    # Extract payoff parameters needed for path simulation
    S0 = payoff_kwargs["S0"]
    T = payoff_kwargs["T"]
    r = payoff_kwargs["r"]
    sigma = payoff_kwargs["sigma"]
    nsteps = int(T / h)

    S = np.zeros((nsamp, nsteps + 1))
    S[:, 0] = S0
    dW = np.random.standard_normal((nsamp, nsteps))
    S = batch_simulate_path(S, r, h, sigma, dW, nsteps, nsamp)
    payoffs_vec = np.vectorize(
        lambda s: payoff(s, h, **payoff_kwargs), signature="(n,m) ->(n)"
    )
    payoffs = payoffs_vec(S)

    result = {
        "esp": np.mean(payoffs),
        "var": np.var(payoffs, ddof=1) / nsamp,
    }

    return result


def coarse_fine_mc(
    nsamp: int, h_coarse: float, payoff: Callable, **payoff_kwargs
) -> Dict[str, float]:

    # Extract payoff parameters needed for path simulation
    S0 = payoff_kwargs["S0"]
    T = payoff_kwargs["T"]
    r = payoff_kwargs["r"]
    sigma = payoff_kwargs["sigma"]

    # define the finer grid to estimate the bias:
    h_fine = h_coarse / 2
    M_coarse = int(T / h_coarse)
    M_fine = 2 * M_coarse

    S_fine = np.zeros((nsamp, M_fine + 1))
    S_fine[:, 0] = S0

    S_coarse = np.zeros((nsamp, M_coarse + 1))
    S_coarse[:, 0] = S0

    dW_fine = np.random.standard_normal(size=(nsamp, M_fine))
    S_fine = batch_simulate_path(S_fine, r, h_fine, sigma, dW_fine, M_fine, nsamp)

    payoff_vec = np.vectorize(
        lambda s, h: payoff(s, h, **payoff_kwargs), signature="(n,m),() ->(n)"
    )
    payoffs_fine = payoff_vec(S_fine, h_fine)

    # dW_coarse = np.add.reduceat(dW_fine, np.arange(0, M_fine, 2), axis=1) / np.sqrt(2.0)
    dW_coarse = (dW_fine[:, 0::2] + dW_fine[:, 1::2]) / np.sqrt(2.0)
    S_coarse = batch_simulate_path(
        S_coarse, r, h_coarse, sigma, dW_coarse, M_coarse, nsamp
    )

    payoffs_coarse = payoff_vec(S_coarse, h_coarse)

    result = {
        "esp_coarse": np.mean(payoffs_coarse),
        "esp_fine": np.mean(payoffs_fine),
        "var_coarse": np.var(payoffs_coarse, ddof=1) / nsamp,
        "var_fine": np.var(payoffs_fine, ddof=1) / nsamp,
        "esp_diff": np.mean(payoffs_fine - payoffs_coarse),
        "var_diff": np.var(payoffs_fine - payoffs_coarse, ddof=1) / nsamp,
        "bias": abs(np.mean(payoffs_coarse) - np.mean(payoffs_fine)),
    }

    return result


def two_level_mc(
    nsamp0: int, nsamp1: int, h_coarse: float, payoff: Callable, **payoff_kwargs
) -> Dict[str, float]:
    """
    Perform a two-level Monte Carlo estimation.
    Parameters:
    nsamp0 (int): Number of samples for the coarse level.
    nsamp1 (int): Number of samples for the fine level.
    h_coarse (float): Coarse level discretization parameter.
    payoff (Callable): Payoff function to be evaluated.
    **payoff_params: Additional parameters to be passed to the payoff function.
    Returns:
    Dict[str, float]: A dictionary containing the estimated expected value ('esp') and variance ('var').
    """

    result_level_0 = standard_mc(nsamp0, h_coarse, payoff, **payoff_kwargs)

    # level 1 estimate
    result_level_1 = coarse_fine_mc(nsamp1, h_coarse, payoff, **payoff_kwargs)

    result = {
        "esp": result_level_0["esp"] + result_level_1["esp_diff"],
        "var": result_level_0["var"] + result_level_1["var_diff"],
        "esp0": result_level_0["esp"],
        "var0": result_level_0["var"],
        "esp01": result_level_1["esp_diff"],
        "var01": result_level_1["var_diff"],
    }

    return result


def mlmc(
    nsamps: List[int], h_coarse: float, payoff: Callable, **payoff_kwargs
) -> Dict[str, float]:
    """
    Perform a multi-level Monte Carlo estimation.
    Parameters:
    nsamp0 (int): Number of samples for the coarse level.
    h_coarse (float): Coarse level discretization parameter.
    payoff (Callable): Payoff function to be evaluated.
    **payoff_params: Additional parameters to be passed to the payoff function.
    Returns:
    Dict[str, float]: A dictionary containing the estimated expected value ('esp') and variance ('var').
    """

    h_values = [h_coarse / 2**i for i in range(len(nsamps))]

    level_means = np.zeros(len(nsamps))
    level_vars = np.zeros(len(nsamps))

    for l, (h, nsamp) in enumerate(zip(h_values, nsamps)):
        level_result = coarse_fine_mc(nsamp, h, payoff, **payoff_kwargs)

        if l == 0:
            level_means[l] = level_result["esp_coarse"]
            level_vars[l] = level_result["var_coarse"]

        else:
            level_means[l] = level_result["esp_diff"]
            level_vars[l] = level_result["var_diff"] #this is Var(Yl - Yl-1)/nsamp

    result = {
        "esp": np.sum(level_means),
        "var": np.sum(level_vars), #this is Var(Yl - Yl-1)/nsamp
    }

    return result
