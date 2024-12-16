import numpy as np


def asian_option(S: np.ndarray, h: float, r: float, K: float, **kwargs) -> float:
    """
    Calculate the payoff of an Asian option using geometric Brownian motion.

    Parameters:
    S (np.ndarray): Array of stock prices. Path of a simulation from the Euler-Maruyama scheme.
    r (float): Risk-free interest rate.
    K (float): Strike price of the option.
    h (float): Time step size.

    Returns:
    float: The payoff of the Asian option.
    """

    Sbar = np.trapezoid(S, dx=h)
    temp = np.stack([Sbar - K, np.zeros_like(Sbar)], axis=1)
    return np.max(temp, axis=1) * np.exp(-r)


def barrier_call_option(S: np.ndarray,  h: float, r: float, K: float, Smax: float, **kwargs) -> float:
    """
    Calculate the payoff of a barrier call option using geometric Brownian motion.

    Parameters:
    S (np.ndarray): Array of stock prices. Path of a simulation from the Euler-Maruyama scheme.
    Smax (float): Barrier level.
    r (float): Risk-free interest rate.
    K (float): Strike price of the option.
    h (float): Time step size.

    Returns:
    float: The payoff of the barrier call option.
    """
    if np.any(S > Smax):
        return 0
    # FIXME: Vectorize this calculation, cf. Asian option
    return max(S[-1] - K, 0) * np.exp(-r)