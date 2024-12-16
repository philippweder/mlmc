import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from mlmc.core.estimators import coarse_fine_mc
from mlmc.core.payoff import asian_option

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
    nsamp_min: int,
    nsamp_max: int,
    nsamp_base: int,
    nsteps_coarse: int,
    out_dir: Path,
    nseeds: int = 1,
) -> None:

    for seed in tqdm(range(nseeds), desc="Scanning seeds", total=nseeds):

        np.random.seed(seed + 9434)  # Set seed for reproducibility

        h_coarse = payoff_params["T"] / nsteps_coarse

        nsamp_start = int(np.log(nsamp_min) / np.log(nsamp_base))
        nsamp_end = int(np.ceil(np.log(nsamp_max) / np.log(nsamp_base)))
        nsamp_values = np.logspace(
            nsamp_start,
            nsamp_end,
            num=nsamp_end - nsamp_start + 1,
            base=nsamp_base,
            dtype=int,
        )
        biases = np.zeros(len(nsamp_values))
        means_coarse = np.zeros(len(nsamp_values))
        variances_coarse = np.zeros(len(nsamp_values))
        variances_fine = np.zeros(len(nsamp_values))
        variances_diff = np.zeros(len(nsamp_values))

        for i, nsamp in enumerate(
            tqdm(nsamp_values, desc="Scanning sample sizes", total=len(nsamp_values))
        ):
            result = coarse_fine_mc(nsamp, h_coarse, asian_option, **payoff_params)
            biases[i] = result["bias"]
            means_coarse[i] = result["esp_coarse"]
            variances_coarse[i] = result["var_coarse"]
            variances_fine[i] = result["var_fine"]
            variances_diff[i] = result["var_diff"]

        df = pd.DataFrame(
            {
                "nsamp": nsamp_values,
                "bias": biases,
                "mean_coarse": means_coarse,
                "variance_coarse": variances_coarse,
                "variance_fine": variances_fine,
                "variance_diff": variances_diff,
            }
        )

        out_path = (
            out_dir / f"nsamp_variation_nsteps_coarse={nsteps_coarse}_seed={seed}.csv"
        )

        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument("--nsamp_min", "-min", type=int, default=10, help="Minimum number of samples.")
    parser.add_argument("--nsamp_max", "-max", type=int, default=100_000, help="Maximum number of samples.")
    parser.add_argument("--nsamp_base", "-b", type=int, default=10, help="Base for the number of samples.")
    parser.add_argument("--nseeds", "-s", type=int, default=1, help="Number of seeds.")
    parser.add_argument(
        "--nsteps_coarse",
        type=int,
        default=200,
        help="Number of steps for the coarse grid",
    )
    args = parser.parse_args()

    main(
        args.nsamp_min,
        args.nsamp_max,
        args.nsamp_base,
        args.nsteps_coarse,
        DATA_DIR,
        args.nseeds,
    )
