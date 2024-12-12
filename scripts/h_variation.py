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


def main(h_min: float, h_max: float, nh: int, nsamp: int, out_path: Path) -> None:
    np.random.seed(9434)  # Set seed for reproducibility

    h_values = np.linspace(h_min, h_max, nh)
    biases = np.zeros(len(h_values))
    means_coarse = np.zeros(len(h_values))
    variances_coarse = np.zeros(len(h_values))
    means_diff = np.zeros(len(h_values))
    variances_diff = np.zeros(len(h_values))

    for i, h in enumerate(
        tqdm(h_values, desc="Scanning time steps", total=len(h_values))
    ):
        result = standard_mc(S0, r, sigma, K, T, nsamp, h)
        biases[i] = result["bias"]
        means_coarse[i] = result["esp_coarse"]
        variances_coarse[i] = result["var_coarse"]
        means_diff[i] = result["esp_diff"]
        variances_diff[i] = result["var_diff"]

    df = pd.DataFrame(
        {
            "h": h_values,
            "bias": biases,
            "mean_coarse": means_coarse,
            "variance_coarse": variances_coarse,
            "mean_diff": means_diff,
            "variance_diff": variances_diff,
        }
    )

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the time step."
    )
    parser.add_argument("--h_min", type=float, default=0.01, help="Minimal time step")
    parser.add_argument("--h_max", type=float, default=0.1, help="Maximal time step")
    parser.add_argument("--nh", type=int, default=10, help="Number of time steps")
    parser.add_argument("--nsamp", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()
    out_path = DATA_DIR / f"h_variation-nsamp={args.nsamp}.csv"

    main(args.h_min, args.h_max, args.nh, args.nsamp, out_path)
