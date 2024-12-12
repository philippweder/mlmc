# function standard MC on N realisations, for time step h (which will be compared
# to time step h/2 for the bias):
import numpy as np
from tqdm import tqdm
from typing import Dict


def standard_mc(
    S0: float, r: float, sigma: float, K: float, T: float, nsamp: int, h_coarse: float
) -> Dict[str, float]:
    """
    Perform standard Monte Carlo simulation for an Asian option using geometric Brownian motion.

    Parameters:
    S0 (float): Initial stock price.
    r (float): Risk-free interest rate.
    sigma (float): Volatility of the stock.
    K (float): Strike price of the option.
    T (float): Time to maturity of the option.
    nsamp (int): Number of Monte Carlo samples.
    h_coarse (float): Time step size for the coarse grid.

    Returns:
    Dict[str, float]: A dictionary containing the results
    """

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
    for n in tqdm(range(nsamp), desc="Simulating Paths"):  # Add progress bar for the loop
        S_fine = np.zeros(
            M_fine + 1
        )  # Array to store the asset prices for a single path
        S_fine[0] = S0  # Initial stock price
        S_coarse = np.zeros(M_coarse + 1)
        S_coarse[0] = S0

        # Obtain the Brownian increments on the fine grid
        dW_fine = np.random.normal(0, np.sqrt(h_fine), size=M_fine)

        # Simulate the path using the Euler-Maruyama scheme, first on the finer grid
        for m in range(M_fine):
            # Update stock price using Euler-Maruyama scheme
            S_fine[m + 1] = (
                S_fine[m] + r * S_fine[m] * h_fine + sigma * S_fine[m] * dW_fine[m]
            )

        # Sum the Brownian increments to create coarser grid increments
        dW_coarse = np.add.reduceat(dW_fine, np.arange(0, M_fine, 2))

        # Simulate the path now on the coarse grid
        for m in range(M_coarse):
            S_coarse[m + 1] = (
                S_coarse[m]
                + r * S_coarse[m] * h_coarse
                + sigma * S_coarse[m] * dW_coarse[m]
            )

        # Approximate the arithmetic average price, for fine and coarse grids
        Sbar_fine = h_fine * np.sum((S_fine[:-1] + S_fine[1:]) / 2)
        Sbar_coarse = h_coarse * np.sum((S_coarse[:-1] + S_coarse[1:]) / 2)

        # Compute the discounted payoff for the Asian option
        payoffs_fine[n] = np.exp(-r) * max(0, Sbar_fine - K)
        payoffs_coarse[n] = np.exp(-r) * max(0, Sbar_coarse - K)

    result = {
        "esp_coarse": np.mean(payoffs_coarse),
        "esp_fine": np.mean(payoffs_fine),
        "var_coarse": np.var(payoffs_coarse, ddof=1)
        / nsamp,  # TODO: Why do we divide by N and why ddof = 1?
        "var_fine": np.var(payoffs_fine, ddof=1)
        / nsamp,  # TODO: Why do we divide by N and why ddof = 1?
        # quantity below would be E[Y_L] in the statement
        "esp_diff": np.mean(payoffs_fine - payoffs_coarse),
        "var_diff": np.var(
            payoffs_fine - payoffs_coarse, ddof=1
        ),  # TODO: Why do we divide by N and why ddof = 1?
        "bias": abs(np.mean(payoffs_coarse) - np.mean(payoffs_fine)),
    }

    return result
