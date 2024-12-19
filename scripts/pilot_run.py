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

DATA_DIR = Path("../data/asian_option/pilot_run")
DATA_DIR.mkdir(exist_ok=True, parents=True)

payoff_params = {
    "T": 1,  # Time to maturity
    "r": 0.05,  # Risk-free interest rate
    "sigma": 0.2,  # Volatility
    "K": 1,  # Strike price
    "S0": 1,  # Initial stock price
}


def main(
    nsamp_pilot: int,
    nlevels_pilot: int,
    out_dir: Path,
):
    # np.random.seed(9434)
    np.random.seed(9336)
    h_coarse = 0.2  # this value for h0 is forced by the statement of the project.

    # to know the proportionality factors E0 and V0
    result = mlmc_pilot(
        nlevels_pilot, nsamp_pilot, h_coarse, asian_option, **payoff_params
    )

    df = pd.DataFrame(
        {
            "biases": result["biases"],
            "variances": result["variances"],  # this is Var(Yl - Yl+1)
            "E0": result["E0"],
            "V0": result["V0"],
            "nlevels_pilot": nlevels_pilot,  # not the optimal number of levels, but the one
            # used for the pilot run
            "nsamp_pilot": nsamp_pilot,  # idem
        }
    )

    out_path = (
        out_dir
        / f"pilot_mlmc_h_coarse={h_coarse}_nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to produce pilot run datas to plot and obtain E0 and V0"
    )

    parser.add_argument(
        "--nsamp_pilot",
        type=int,
        default=10_000,
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
        args.nsamp_pilot,
        args.nlevels_pilot,
        DATA_DIR,
    )
