import numpy as np
from typing import Dict, List, Tuple

from mlmc.core.options import Option
from mlmc.core.sde import batch_simulate_path


def standard_mc(nsamp: int, h: float, option: Option) -> Dict[str, float]:
    """
    Perform a standard Monte Carlo simulation for option pricing.
    Parameters:
    nsamp (int): Number of samples (simulated paths).
    h (float): Time step size.
    option (Option): An instance of the Option class containing option parameters.
    Returns:
    Dict[str, float]: A dictionary containing the estimated expected payoff ('esp') and the variance of the payoff ('var').
    """

    nsteps = int(option.T / h)

    # initialize paths and Brownian increments
    S = np.zeros((nsamp, nsteps + 1))
    S[:, 0] = option.S0
    dW = np.random.standard_normal((nsamp, nsteps))  # N(0,1)

    S = batch_simulate_path(S, option.r, h, option.sigma, dW, nsteps, nsamp)
    payoffs = option.payoff(S, h)
    result = {
        "esp": np.mean(payoffs),
        "var": np.var(payoffs, ddof=1) / nsamp,
    }

    return result


def is_mc_drift_in_bm(
    nsamp: int, h: float, option: Option, R: float
) -> Dict[str, float]:
    """
    Simulates Monte Carlo paths for a given option under a higher interest rate and computes the expected payoff and variance.
    (Importance Sampling estimator)
    Parameters:
    nsamp (int): Number of samples (paths) to simulate.
    h (float): Time step size.
    option (Option): Option object containing parameters such as initial stock price (S0), maturity (T), volatility (sigma), risk-free rate (r), and payoff function.
    R (float): New interest rate, higher than the risk-free rate (r).
    Returns:
    Dict[str, float]: A dictionary containing:
        - "esp": The expected payoff of the option under the new interest rate.
        - "var": The variance of the expected payoff divided by the number of samples.
    """

    nsteps = int(option.T / h)

    S = np.zeros((nsamp, nsteps + 1))
    S[:, 0] = option.S0
    phi = (R - option.r) / option.sigma  # R is the new interest rate, higher than r
    dW_drift = np.random.normal(loc=phi * h, scale=np.sqrt(h), size=(nsamp, nsteps))

    # simulate paths with higher interest rate R, this is embedded into dW_drift
    S = batch_simulate_path(
        S,
        option.r,
        h,
        option.sigma,
        dW_drift,
        nsteps,
        nsamp,
        unnormalized_noise=True,
    )

    payoffs = option.payoff(S, h)

    # compute likelihood ratio
    w = np.exp(0.5 * nsteps * h * phi**2 - phi * np.sum(dW_drift, axis=1))

    result = {
        "esp": np.mean(payoffs * w),
        "var": np.var(payoffs * w, ddof=1) / nsamp,
    }

    return result



def mlmc_level(nsamp: int, h: float, option: Option) -> Tuple[float, float]:
    """
    Simulates paths for the Multilevel Monte Carlo (MLMC) method and computes payoffs at two levels of discretization.
    Parameters:
    nsamp (int): Number of samples to simulate.
    h (float): Time step size for the coarse level.
    option (Option): An instance of the Option class containing the option parameters (e.g., initial stock price S0, risk-free rate r, volatility sigma, and maturity T).
    Returns:
    Tuple[float, float]: A tuple containing the payoffs for the coarse and fine levels of discretization.
    """

    h_fine = h / 2
    nsteps_coarse = int(option.T / h)
    nsteps_fine = 2 * nsteps_coarse

    S_fine = np.zeros((nsamp, nsteps_fine + 1))
    S_fine[:, 0] = option.S0

    S_coarse = np.zeros((nsamp, nsteps_coarse + 1))
    S_coarse[:, 0] = option.S0

    dW_fine = np.random.standard_normal(size=(nsamp, nsteps_fine))
    S_fine = batch_simulate_path(
        S_fine, option.r, h_fine, option.sigma, dW_fine, nsteps_fine, nsamp
    )

    payoffs_fine = option.payoff(S_fine, h_fine)

    dW_coarse = (dW_fine[:, 0::2] + dW_fine[:, 1::2]) / np.sqrt(2.0)
    S_coarse = batch_simulate_path(
        S_coarse, option.r, h, option.sigma, dW_coarse, nsteps_coarse, nsamp
    )
    payoffs_coarse = option.payoff(S_coarse, h)

    return payoffs_coarse, payoffs_fine


def coarse_fine_mc(nsamp: int, h_coarse: float, option: Option) -> Dict[str, float]:
    """
    Perform coarse and fine Monte Carlo estimations and compute various statistics.
    Parameters:
    nsamp (int): Number of samples to generate.
    h_coarse (float): Coarse level parameter.
    option (Option): Option object containing parameters for the Monte Carlo simulation.
    Returns:
    Dict[str, float]: A dictionary containing the following keys:
        - "esp_coarse": Mean of the coarse payoffs.
        - "esp_fine": Mean of the fine payoffs.
        - "var_coarse": Variance of the coarse payoffs divided by the number of samples.
        - "var_fine": Variance of the fine payoffs divided by the number of samples.
        - "esp_diff": Mean of the difference between fine and coarse payoffs.
        - "var_diff": Variance of the difference between fine and coarse payoffs divided by the number of samples.
        - "bias": Absolute difference between the mean of coarse and fine payoffs.
        - "sum_coarse": Sum of the coarse payoffs.
        - "sum_coarse2": Sum of the squares of the coarse payoffs.
        - "sum_fine": Sum of the fine payoffs.
        - "sum_diff": Sum of the differences between fine and coarse payoffs.
        - "sum_diff2": Sum of the squares of the differences between fine and coarse payoffs.
    """

    payoffs_coarse, payoffs_fine = mlmc_level(nsamp, h_coarse, option)

    result = {
        "esp_coarse": np.mean(payoffs_coarse),
        "esp_fine": np.mean(payoffs_fine),
        "var_coarse": np.var(payoffs_coarse, ddof=1) / nsamp,
        "var_fine": np.var(payoffs_fine, ddof=1) / nsamp,
        "esp_diff": np.mean(payoffs_fine - payoffs_coarse),
        "var_diff": np.var(payoffs_fine - payoffs_coarse, ddof=1) / nsamp,
        "bias": abs(np.mean(payoffs_coarse) - np.mean(payoffs_fine)),
        "sum_coarse": np.sum(payoffs_coarse),
        "sum_coarse2": np.sum(payoffs_coarse**2),
        "sum_fine": np.sum(payoffs_fine),
        "sum_diff": np.sum(payoffs_fine - payoffs_coarse),
        "sum_diff2": np.sum((payoffs_fine - payoffs_coarse) ** 2),
    }

    return result


def two_level_mc(
    nsamp0: int, nsamp1: int, h_coarse: float, option: Option
) -> Dict[str, float]:
    """
    Perform a two-level Monte Carlo estimation.
    Parameters:
    nsamp0 (int): Number of samples for the coarse level.
    nsamp1 (int): Number of samples for the fine level.
    h_coarse (float): Coarse level step size.
    option (Option): Configuration options for the Monte Carlo simulation.
    Returns:
    Dict[str, float]: A dictionary containing the following keys:
        - "esp": Estimated expected value.
        - "var": Estimated variance.
        - "esp0": Expected value from the coarse level.
        - "var0": Variance from the coarse level.
        - "esp01": Difference in expected value between coarse and fine levels.
        - "var01": Difference in variance between coarse and fine levels.
    """
    
    result_level_0 = standard_mc(nsamp0, h_coarse, option)
    result_level_1 = coarse_fine_mc(nsamp1, h_coarse, option)

    result = {
        "esp": result_level_0["esp"] + result_level_1["esp_diff"],
        "var": result_level_0["var"] + result_level_1["var_diff"],
        "esp0": result_level_0["esp"],
        "var0": result_level_0["var"],
        "esp01": result_level_1["esp_diff"],
        "var01": result_level_1["var_diff"],
    }

    return result


def mlmc(nsamps: List[int], h_coarse: float, option: Option) -> Dict[str, float]:
    """
    Perform Multilevel Monte Carlo (MLMC) estimation.
    Parameters:
    nsamps (List[int]): A list of integers where each integer represents the number of samples at each level.
    h_coarse (float): The coarsest level step size.
    option (Option): An object containing options for the MLMC estimation.
    Returns:
    Dict[str, float]: A dictionary containing the estimated expected value ('esp') and the estimated variance ('var').
    Notes:
    - The function calculates the step sizes for each level by halving the coarsest level step size.
    - It computes the mean and variance of the payoffs at each level.
    - The final result is the sum of the means and variances across all levels.
    """

    h_values = [h_coarse / 2**i for i in range(len(nsamps))]
    level_means = np.zeros(len(nsamps))
    level_vars = np.zeros(len(nsamps))

    for l, (h, nsamp) in enumerate(zip(h_values, nsamps)):
        pc, pf = mlmc_level(nsamp, h, option)
        # the above returns payoff coarse (pc) and payoff fine (pf)
        # for hcoarse = h, and h fine = h/2.

        if l == 0:
            level_means[l] = np.mean(pc)
            level_vars[l] = np.var(pc, ddof=1) / nsamp

        else:
            level_means[l] = np.mean(pf - pc)
            level_vars[l] = np.var(pf - pc, ddof=1) / nsamp

    result = {
        "esp": np.sum(level_means),
        "var": np.sum(level_vars),
    }

    return result
