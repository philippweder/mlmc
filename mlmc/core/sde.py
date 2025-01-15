import numpy as np
from numba import njit, prange

# speed up computations using numba
@njit(parallel=True)
def batch_simulate_path(
    S: np.ndarray,
    r: float,
    h: float,
    sigma: float,
    dW: np.ndarray,
    nsteps: int,
    nsamp: int,
    unnormalized_noise: bool = False,  # Set to true if you need to use dW \~ N(0,1), e.g. for importance sampling
) -> np.ndarray:
    """
    Simulates paths of a stochastic process using the Euler-Maruyama scheme.
    Parameters:
    S (np.ndarray): Initial stock prices, shape (nsamp, nsteps+1).
    r (float): Risk-free interest rate.
    h (float): Time step size.
    sigma (float): Volatility of the stock.
    dW (np.ndarray): Brownian increments, shape (nsamp, nsteps).
    nsteps (int): Number of time steps.
    nsamp (int): Number of sample paths.
    unnormalized_noise (bool, optional): If True, use unnormalized noise dW ~ N(0,1). Default is False.
    Returns:
    np.ndarray: Simulated stock prices, shape (nsamp, nsteps+1).
    """

    nsamp = S.shape[0]
    sqrt_h = np.sqrt(h)

    for n in prange(nsamp):
        for m in range(nsteps):
            # Update stock price using Euler-Maruyama scheme
            if unnormalized_noise:
                S[n, m + 1] = (1.0 + r * h + sigma * dW[n, m]) * S[n, m]
            else:
                S[n, m + 1] = (1.0 + r * h + sigma * sqrt_h * dW[n, m]) * S[n, m]

    return S