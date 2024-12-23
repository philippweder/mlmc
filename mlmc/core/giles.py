from typing import Callable, Dict, Union
import numpy as np

from mlmc.core.estimators import coarse_fine_mc
from mlmc.core.helpers import fit_exponential_decay
from mlmc.core.payoff import asian_option
from mlmc.core.options import Option, AsianOption


def mlmc_giles(
    eps: float,
    nsamp_init: int,
    h_init: float,
    option: Option,
    alpha_init: float = 0,
    beta_init: float = 0,
    gamma: float = 1,
) -> Dict[str, Union[float, np.ndarray]]:
    alpha = np.max(0, alpha_init)
    beta = np.max(0, beta_init)

    L = 2
    nsamp_levels = np.zeros(L + 1, dtype=int)
    mean_sums = np.zeros(L + 1)
    square_sums = np.zeros(L + 1)
    dnsamp_levels = np.full(L + 1, nsamp_init)

    # loop until no additional samples are needed
    while np.sum(dnsamp_levels) > 0:

        # update the sums on each level
        for l in range(L + 1):
            if dnsamp_levels[l] > 0:
                hl = h_init * 2**l
                result = coarse_fine_mc(dnsamp_levels[l], hl, option)
                nsamp_levels[l] += dnsamp_levels[l]
                if l:
                    mean_sums[l] += result["sum_diff"]
                    square_sums[l] += result["sum_diff2"]
                else:
                    mean_sums[l] += result["sum_coarse"]
                    square_sums[l] += result["sum_coarse2"]

        # compute bias and variance
        biases = np.abs(mean_sums) / nsamp_levels
        variances = np.maximum(
            0.0,
            square_sums / (nsamp_levels - 1)
            - (nsamp_levels - 2) / (nsamp_levels - 1) * biases**2,
        )

        for l in range(3, L + 1):
            biases[l] = np.maximum(biases[l], 0.5 * biases[l - 1] / 2**alpha)
            variances[l] = np.maximum(variances[l], 0.5 * variances[l - 1] / 2**beta)

        # compute alpha and beta if not given using linear regression
        if alpha_init <= 0:
            _, alpha = fit_exponential_decay(np.arange(1, L + 1), biases[1:])
            alpha = np.maximum(0.5, alpha)
        if beta_init <= 0:
            _, beta = fit_exponential_decay(np.arange(1, L + 1), variances[1:])
            beta = np.maximum(0.5, beta)

        # set optimal number of additional samples
        costs = 2 ** (gamma * np.arange(L + 1))
        nsamp_optimal = np.ceil(
            2 * np.sqrt(variances / costs) * np.sum(np.sqrt(variances * costs)) / eps**2
        )
        dnsamp_levels = np.maximum(0, nsamp_optimal - nsamp_levels).astype(int)

        if np.sum(dnsamp_levels > 0.01 * nsamp_levels) == 0:
            remainder_levels = np.arange(L - 2, L + 1)
            remainder = np.max(
                biases[remainder_levels]
                * 2 ** (alpha * remainder_levels)
                / (2**alpha - 1)
            )
            if remainder > eps / np.sqrt(2):
                L = L + 1
                variances = np.append(variances, variances[-1] / 2**beta)
                nsamp_levels = np.append(nsamp_levels, 0)
                mean_sums = np.append(mean_sums, 0)
                square_sums = np.append(square_sums, 0)

                costs = 2 ** (gamma * np.arange(L + 1))
                nsamp_optimal = np.ceil(
                    2
                    * np.sqrt(variances / costs)
                    * np.sum(np.sqrt(variances * costs))
                    / eps**2
                )
                dnsamp_levels = np.maximum(0, nsamp_optimal - nsamp_levels).astype(int)

    # compute the final estimate
    return {
        "esp": np.sum(mean_sums / nsamp_levels),
        "nsamp_levels": nsamp_levels,
        "nlevels": L,
    }


if __name__ == "__main__":
    h_init = 0.2
    option = AsianOption()
    result = mlmc_giles(1e-5, 1000, h_init, option)
    for k, v in result.items():
        print(f"{k}: {v}")
