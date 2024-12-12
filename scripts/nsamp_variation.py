import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from mlmc.core.standard import standard_mc

DATA_DIR = Path("../data/asian_option")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Define constants for the problem
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
S0 = 1  # Initial stock price
K = 1  # Strike price
T = 1  # Time to maturity

def main(nsamp_min: int, nsamp_max: int, nsamp_step: int, nsteps_coarse: int, out_path: Path) -> None:
    np.random.seed(9434) # Set seed for reproducibility

    h_coarse = T / nsteps_coarse
    nsamp_values = np.arange(nsamp_min, nsamp_max + 1, nsamp_step)
    biases = np.zeros(len(nsamp_values))
    means_coarse = np.zeros(len(nsamp_values))
    variances_coarse = np.zeros(len(nsamp_values))

    for i, nsamp in enumerate(
        tqdm(nsamp_values, desc="Scanning sample sizes", total=len(nsamp_values))
    ):
        result = standard_mc(S0, r, sigma, K, T, nsamp, h_coarse)
        biases[i] = result["bias"]
        means_coarse[i] = result["esp_coarse"]
        variances_coarse[i] = result["var_coarse"]

    df = pd.DataFrame(
        {
            "nsamp": nsamp_values,
            "bias": biases,
            "mean_coarse": means_coarse,
            "variance_coarse": variances_coarse,
        }
    )

    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsamp_min", type=int, default=100, help="Minimal number of samples"
    )
    parser.add_argument(
        "--nsamp_max", type=int, default=10000, help="Maximal number of samples"
    )
    parser.add_argument(
        "--nsamp_step", type=int, default=10, help="Increment for samples"
    )
    parser.add_argument(
        "--nsteps_coarse",
        type=int,
        default=200,
        help="Number of steps for the coarse grid",
    )
    args = parser.parse_args()
    out_path = DATA_DIR / f"nsamp_variation-nsteps_coarse={args.nsteps_coarse}.csv"
    
    main(args.nsamp_min, args.nsamp_max, args.nsamp_step, args.nsteps_coarse, out_path)
