from typing import Callable, Dict
import numpy as np

from mlmc.core.estimators import coarse_fine_mc


def mlmc_pilot(
    mse: float,
    nlevels: int,
    nsamp: int,
    h_coarse: float,
    payoff: Callable,
    **payoff_kwargs
) -> Dict[str, float]:

    h_values = [h_coarse / 2**i for i in range(nlevels)]

    biases = []
    variances = []

    for h in h_values:
        result = coarse_fine_mc(nsamp, h, payoff, **payoff_kwargs)
        biases.append(result["esp_diff"])
        variances.append(result["var_diff"])

    biases = np.array(biases)
    variances = np.array(variances)

    E0 = np.polyfit(np.arange(nlevels), biases * (2 ** np.arange(nlevels)), 0)[-1]
    V0 = np.polyfit(
        np.arange(nlevels), nsamp * variances * (2 ** np.arange(nlevels)), 0
    )[-1]

    optimal_nlevels = int(np.ceil(np.log2(E0 / np.sqrt(mse))))
    optimal_nsamps = [
        int(np.ceil(2 ** (-l) * (optimal_nlevels + 1) * V0 / mse))
        for l in range(optimal_nlevels)
    ]

    return {
        "biases": biases,
        "variances": variances * nsamp,
        "E0": E0,
        "V0": V0,
        "nlevels": optimal_nlevels,
        "nsamps": optimal_nsamps,
    }
