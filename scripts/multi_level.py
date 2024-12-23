import argparse
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time

from mlmc.core.estimators import mlmc, standard_mc
from mlmc.core.options import AsianOption
from mlmc.core.payoff import asian_option
from mlmc.core.helpers import compute_optimal_samps, mlmc_pilot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/asian_option")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def main(
    eps_val: List[float],
    nsamp_pilot: int,
    nlevels_pilot: int,
    out_dir: Path,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float = 1,
):
    if alpha is None or beta is None:
        est_str = "estimated"
    else:
        est_str = "prescribed"

    np.random.seed(9434)
    option = AsianOption()
    h_coarse = 0.2  # this value for h0 is forced by the statement of the project.

    means_mlmc = np.zeros(len(eps_val))
    variances_mlmc = np.zeros(len(eps_val))
    nlevels = np.zeros(len(eps_val))  # L
    cpu_times = np.zeros(len(eps_val))
    optimal_nsamps_list = []

    means_mc = np.zeros(len(eps_val))
    variances_mc = np.zeros(len(eps_val))
    nsamps_mc = np.zeros(len(eps_val))
    cpu_times_mc = np.zeros(len(eps_val))

    pilot_results = mlmc_pilot(
        nlevels_pilot,
        nsamp_pilot,
        h_coarse,
        option,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    E0 = pilot_results["E0"]
    alpha = pilot_results["alpha"]
    V0 = pilot_results["V0"]
    beta = pilot_results["beta"]
    V_mc = pilot_results["Vfine"]

    logger.info(f"E0: {E0:5e}")
    logger.info(f"V0: {V0:5e}")

    pilot_df = {
        "biases": pilot_results["biases"],
        "variances": pilot_results["variances"],
    }

    fn = (
        out_dir
        / f"mlmc_pilot_nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={est_str}.csv"
    )
    pd.DataFrame(pilot_df).to_csv(fn, index=False)

    pbar = tqdm(enumerate(eps_val), total=len(eps_val), desc="Scanning epsilon values")

    for i, eps in pbar:
        pbar.set_postfix({"eps": eps})
        optimal_L, optimal_nsamps = compute_optimal_samps(
            E0, V0, eps, alpha=alpha, beta=beta
        )
        logger.info(f"alpha: {alpha}, beta: {beta}")
        logger.info(f"Optimal number of levels: {optimal_L}")
        logger.info(f"Optimal number of samples: {optimal_nsamps}")

        # below : run the actual MLMC simulation using optimal Nl and L
        start_cpu_time = time.process_time()
        result = mlmc(optimal_nsamps, h_coarse, option)
        end_cpu_time = time.process_time()

        cpu_times[i] = end_cpu_time - start_cpu_time
        means_mlmc[i] = result["esp"]
        variances_mlmc[i] = result["var"]  # this is var(Yl - Yl-1)/nsamp
        nlevels[i] = optimal_L + 1

        optimal_nsamps_list.append(optimal_nsamps)

        logger.info(f"CPU Time MLMC: {cpu_times[i]:.6f} s")

        nsamp_crude = int(np.ceil(V_mc / eps**2))

        h_crude = h_coarse * (2**optimal_L)
        start_cpu_time = time.process_time()
        result_crude = standard_mc(nsamp_crude, h_crude, option)
        end_cpu_time = time.process_time()

        cpu_times_mc[i] = end_cpu_time - start_cpu_time
        means_mc[i] = result_crude["esp"]
        variances_mc[i] = result_crude["var"]
        nsamps_mc[i] = nsamp_crude

        logger.info(f"Number of samples for MC: {nsamp_crude}")
        logger.info(f"CPU Time MC: {cpu_times_mc[i]:.6f} s")

    df_outputs = pd.DataFrame(
        {
            "eps": eps_val,
            "cpu_time": cpu_times,
            "mean": means_mlmc,  # this is for each level esp(Yl - Yl-1)
            "variance": variances_mlmc,  # this is var(Yl - Yl-1)/nsamp
            "nlevels": nlevels,
            "variance_mc": variances_mc,
            "mean_mc": means_mc,
            "nsamp_mc": nsamps_mc,
            "cpu_time_mc": cpu_times_mc,
        }
    )

    df_outputs["E0"] = E0
    df_outputs["alpha"] = alpha
    df_outputs["V0"] = V0
    df_outputs["beta"] = beta

    out_path = (
        out_dir
        / f"mlmc-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={est_str}.csv"
    )
    df_outputs.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    max_nlevels = int(nlevels.max()) + 1
    optimal_nsamps_array = np.zeros((len(eps_val), max_nlevels))
    for i, optimal_nsamps in enumerate(optimal_nsamps_list):
        optimal_nsamps_array[i, : len(optimal_nsamps)] = optimal_nsamps

    df_nlevels = pd.DataFrame(optimal_nsamps_array)
    df_nlevels.rename(
        {i: f"level_{i}" for i in range(max_nlevels)}, axis=1, inplace=True
    )
    df_nlevels["eps"] = eps_val
    out_path_nlevels = (
        out_dir
        / f"mlmc_nlevels-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={est_str}.csv"
    )
    df_nlevels.to_csv(out_path_nlevels, index=False)
    logger.info(f"Results saved to {out_path_nlevels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the MSE for the MLMC estimator."
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        help="Target accuracies epsilon for which to compute the MLMC estimator",
    )
    parser.add_argument(
        "--nsamp_pilot",
        type=int,
        default=50_000,
        help="Number of samples for the pilot run.",
    )
    parser.add_argument(
        "--nlevels_pilot",
        type=int,
        default=8,
        help="Number of levels for the pilot run.",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=None,
        help="Prescribed decay of biase. Set to none to estimate it.",
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=None,
        help="Prescribed decay of variance. Set to none to estimate it.",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=float,
        default=1.0,
        help="Prescribed growth of computational costs.",
    )
    args = parser.parse_args()

    main(
        args.eps,
        args.nsamp_pilot,
        args.nlevels_pilot,
        DATA_DIR,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
