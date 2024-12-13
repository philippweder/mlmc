# function standard MC on N realisations, for time step h (which will be compared
# to time step h/2 for the bias):
import numpy as np
from tqdm import tqdm
from typing import Dict, Callable, Tuple
from numba import jit


# speed up computations using numba
@jit(nopython=True)
def simulate_path(
    S: np.ndarray, r: float, h: float, sigma: float, dW: np.ndarray, M: int
) -> np.ndarray:
    """
    Simulates the path of a stock price using the Euler-Maruyama scheme.

    Parameters:
    S (numpy.ndarray): Array of stock prices, where S[0] is the initial stock price.
    r (float): Risk-free interest rate.
    h (float): Time step size.
    sigma (float): Volatility of the stock.
    dW (numpy.ndarray): Array of increments of the Wiener process.
    M (int): Number of time steps.

    Returns:
    numpy.ndarray: Array of simulated stock prices.
    """

    for m in range(M):
        # Update stock price using Euler-Maruyama scheme
        S[m + 1] = S[m] + r * S[m] * h + sigma * S[m] * dW[m]
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

    # Pre-allocate storage for the payoffs
    payoffs = np.zeros(nsamp)

    # Simulate N independent paths of the geometric Brownian motion
    for n in tqdm(range(nsamp), desc="Simulating Paths"):
        S = np.zeros(nsteps + 1)
        S[0] = S0
        dW = np.random.normal(0, np.sqrt(h), size=nsteps)
        S = simulate_path(S, r, h, sigma, dW, nsteps)
        payoffs[n] = payoff(S, h, **payoff_kwargs)

    result = {
        "esp": np.mean(payoffs),
        "var": np.var(payoffs, ddof=1) / nsamp,
    }

    return result


def coarse_fine_mc(
    nsamp: int, h_coarse: float, payoff: Callable, **payoff_kwargs
) -> Dict[str, float]:
    """
    Perform standard Monte Carlo simulation for an Asian option using geometric Brownian motion.

    Parameters:
    S0 (float): Initial stock price.
    sigma (float): Volatility of the stock.
    T (float): Time to maturity of the option.
    nsamp (int): Number of Monte Carlo samples.
    h_coarse (float): Time step size for the coarse grid.

    Returns:
    Dict[str, float]: A dictionary containing the results
    """

    # Extract payoff parameters needed for path simulation
    S0 = payoff_kwargs["S0"]
    T = payoff_kwargs["T"]
    r = payoff_kwargs["r"]
    sigma = payoff_kwargs["sigma"]

    # define the finer grid to estimate the bias:
    h_fine = h_coarse / 2
    M_coarse = int(T / h_coarse)
    M_fine = int(T / h_fine)

    # Pre-allocate storage for the payoffs
    payoffs_coarse = np.zeros(
        nsamp
    )  # Array to store the payoffs for each path, coarse time grid
    payoffs_fine = np.zeros(
        nsamp
    )  # Array to store the payoffs for each path, finer time grid

    # Simulate N independent paths of the geometric Brownian motion
    for n in tqdm(
        range(nsamp), desc="Simulating Paths"
    ):  # Add progress bar for the loop
        S_fine = np.zeros(
            M_fine + 1
        )  # Array to store the asset prices for a single path
        S_fine[0] = S0  # Initial stock price
        S_coarse = np.zeros(M_coarse + 1)
        S_coarse[0] = S0

        # Obtain the Brownian increments on the fine grid
        dW_fine = np.random.normal(0, np.sqrt(h_fine), size=M_fine)

        # Simulate the path using the Euler-Maruyama scheme, first on the finer grid
        S_fine = simulate_path(S_fine, r, h_fine, sigma, dW_fine, M_fine)

        # Sum the Brownian increments to create coarser grid increments
        dW_coarse = np.add.reduceat(dW_fine, np.arange(0, M_fine, 2))

        # Simulate the path now on the coarse grid
        S_coarse = simulate_path(S_coarse, r, h_coarse, sigma, dW_coarse, M_coarse)

        # Calculate the payoff for the fine and coarse paths
        payoffs_fine[n] = payoff(S_fine, h_fine, **payoff_kwargs)
        payoffs_coarse[n] = payoff(S_coarse, h_coarse, **payoff_kwargs)

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
    nsamp0: int, nsamp1: int, h_coarse: float, payoff: Callable, **payoff_params
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

    result_level_0 = standard_mc(nsamp0, h_coarse, payoff, **payoff_params)

    # level 1 estimate
    result_level_1 = coarse_fine_mc(nsamp1, h_coarse, payoff, **payoff_params)

    result = {
        "esp": result_level_0["esp"] + result_level_1["esp_diff"],
        "var": result_level_0["var"] + result_level_1["var_diff"],
    }

    return result
