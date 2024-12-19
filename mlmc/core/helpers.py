from typing import Callable, Dict
import numpy as np

from mlmc.core.estimators import coarse_fine_mc


def mlmc_pilot(
    nlevels: int,  # initial number of levels
    nsamp: int,  # initial number of samples (same for each level)
    h_coarse: int,
    payoff: Callable,
    **payoff_kwargs
) -> Dict[str, float]:

    h_values = [h_coarse / 2**i for i in range(nlevels)]

    biases = []
    variances = []

    for h in h_values:
        # coarse_fine_mc estimates difference between fine-level (2h) and coarse-level(h)
        result = coarse_fine_mc(nsamp, h, payoff, **payoff_kwargs)
        biases.append(result["esp_diff"])
        variances.append(
            result["var_diff"]
        )  # this quantity is Var( Y_fine - Y_coarse)/n_samp

    biases = np.array(biases)
    variances = np.array(variances)

    E0 = np.polyfit(
        np.arange(1, nlevels), biases[1:] * (2 ** np.arange(1, nlevels)), 0
    )[-1]
    V0 = np.polyfit(
        np.arange(1, nlevels),
        variances[1:] * nsamp * (2.0 ** (np.arange(1, nlevels))),
        0,
    )[-1]

    return {
        "biases": biases,
        "variances": variances * nsamp,  # now this is Var(Yfine - Ycoarse)
        "E0": E0,
        "V0": V0,
    }
