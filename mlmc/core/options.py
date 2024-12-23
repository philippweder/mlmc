from abc import ABC, abstractmethod
import numpy as np
from numba import njit


class Option(ABC):
    """
    Abstract base class for financial options.

    Attributes:
        S0 (float): Initial stock price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.

    Methods:
        payoff(S: np.ndarray, h: float) -> np.ndarray:
            Abstract method to calculate the payoff of the option
            for a batch of stock paths.
    """

    def __init__(self, S0: float, T: float, r: float, sigma: float):
        self.S0 = S0  # Initial stock price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the stock

    @abstractmethod
    def payoff(self, S: np.ndarray, h: float, payoffs: np.ndarray) -> float:
        raise NotImplementedError


class AsianOption(Option):

    def __init__(
        self,
        S0: float = 1,
        T: float = 1,
        r: float = 0.05,
        sigma: float = 0.2,
        K: float = 1,
    ):
        super().__init__(S0, T, r, sigma)
        self.K = K  # Strike price

    def payoff(self, S: np.ndarray, h: float) -> np.ndarray:
        payoffs = np.zeros(S.shape[0])
        return self._payoff(S, h, payoffs, self.K, self.r)

    @staticmethod
    @njit
    def _payoff(
        S: np.ndarray, h: float, payoffs: np.ndarray, K: float, r: float
    ) -> np.ndarray:
        for n in range(S.shape[0]):
            Sbar = 0.5 * h * np.sum(S[n, 1:] + S[n, :-1])
            payoffs[n] = max([Sbar - K, 0.0]) * np.exp(-r)

        return payoffs
