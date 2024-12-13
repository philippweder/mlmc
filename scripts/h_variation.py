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
    nsteps_min: int, nsteps_max: int, nsteps_base: int, nsamp: int, out_dir: Path, nseeds: int = 1
) -> None:
    
    for seed in tqdm(range(nseeds), desc="Scanning seeds", total=nseeds):
        np.random.seed(seed + 9434)  # Set seed for reproducibility

        nsteps_start = int(np.log(nsteps_min) / np.log(nsteps_base))
        nsteps_end = int(np.ceil((np.log(nsteps_max) / np.log(nsteps_base))))
        nsteps_values = np.logspace(
            nsteps_start,
            nsteps_end,
            num=nsteps_end - nsteps_start + 1,
            base=nsteps_base,
            dtype=int,
            endpoint=True,
        )

        h_values = payoff_params["T"] / nsteps_values
        biases = np.zeros(len(h_values))
        means_coarse = np.zeros(len(h_values))
        variances_coarse = np.zeros(len(h_values))
        means_diff = np.zeros(len(h_values))
        variances_diff = np.zeros(len(h_values))

        for i, h in enumerate(
            tqdm(h_values, desc="Scanning time steps", total=len(h_values))
        ):
            result = coarse_fine_mc(nsamp, h, asian_option, **payoff_params)
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

        out_path = out_dir / f"h_variation_nsamp={nsamp}_seed={seed}.csv"
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the time step."
    )
    parser.add_argument("--nsteps_min", type=int, default=10, help="Minimal time step")
    parser.add_argument(
        "--nsteps_max", type=int, default=1000, help="Maximal time step"
    )
    parser.add_argument(
        "--nsteps_base", type=int, default=10, help="Basis for time step"
    )
    parser.add_argument("--nsamp", type=int, default=1000, help="Number of samples")
    parser.add_argument("--nseeds", type=int, default=1, help="Number of seeds")
    args = parser.parse_args()

    main(args.nsteps_min, args.nsteps_max, args.nsteps_base, args.nsamp, DATA_DIR, args.nseeds)
