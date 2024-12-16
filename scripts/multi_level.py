import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from mlmc.core.estimators import mlmc
from mlmc.core.helpers import mlmc_pilot
from mlmc.core.payoff import asian_option

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_DIR = Path("../data/asian_option")
DATA_DIR.mkdir(exist_ok=True, parents=True)

payoff_params = {
    "T": 1,  # Time to maturity
    "r": 0.05,  # Risk-free interest rate
    "sigma": 0.2,  # Volatility
    "K": 1,  # Strike price
    "S0": 1,  # Initial stock price
}


def main(
    mse_min: float,
    mse_max: float,
    mse_base: float,
    h_coarse: float,
    nsamp_pilot: int,
    nlevels_pilot: int,
    out_dir: Path,
):
    np.random.seed(9434)

    mse_start = int(np.log(mse_min) / np.log(mse_base))
    mse_end = int(np.floor(np.log(mse_max) / np.log(mse_base)))
    mse_values = np.logspace(
        mse_start,
        mse_end,
        num=mse_end - mse_start + 1,
        base=mse_base,
    )

    means = np.zeros(len(mse_values))
    variances = np.zeros(len(mse_values))
    nlevels = np.zeros(len(mse_values))

    pbar = tqdm(
        enumerate(mse_values), total=len(mse_values), desc="Scanning MSE values"
    )

    for i, mse in pbar:
        pilot = mlmc_pilot(
            mse, nlevels_pilot, nsamp_pilot, h_coarse, asian_option, **payoff_params
        )

        logger.info(f"Optimal number of levels: {pilot['nlevels']}")
        logger.info(f"Optimal number of samples: {pilot['nsamps']}")

        result = mlmc(pilot["nsamps"], h_coarse, asian_option, **payoff_params)

        means[i] = result["esp"]
        variances[i] = result["var"]
        nlevels[i] = pilot["nlevels"]

    df = pd.DataFrame(
        {
            "mse": mse_values,
            "mean": means,
            "variance": variances,
            "nlevels": nlevels,
        }
    )

    out_path = (
        out_dir
        / f"mlmc_h_coarse={h_coarse}_nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the MSE for the MLMC estimator."
    )
    parser.add_argument(
        "--mse_min", type=float, default=1e-10, help="Minimum MSE value."
    )
    parser.add_argument(
        "--mse_max", type=float, default=1e-6, help="Maximum MSE value."
    )
    parser.add_argument(
        "--mse_base", type=float, default=10, help="Base for the MSE values."
    )
    parser.add_argument(
        "--h_coarse",
        type=float,
        default=1,
        help="Coarse level discretization parameter.",
    )
    parser.add_argument(
        "--nsamp_pilot",
        type=int,
        default=20_000,
        help="Number of samples for the pilot run.",
    )
    parser.add_argument(
        "--nlevels_pilot",
        type=int,
        default=5,
        help="Number of levels for the pilot run.",
    )
    args = parser.parse_args()

    main(
        args.mse_min,
        args.mse_max,
        args.mse_base,
        args.h_coarse,
        args.nsamp_pilot,
        args.nlevels_pilot,
        DATA_DIR,
    )
