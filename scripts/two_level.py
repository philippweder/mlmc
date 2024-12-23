import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from mlmc.core.options import AsianOption
from mlmc.core.estimators import coarse_fine_mc, standard_mc, two_level_mc

DATA_DIR = Path("../data/asian_option")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def main(
    nsamp_min: int,
    nsamp_max: int,
    nsamp_base: int,
    nsamp_pilot: int,
    out_dir: Path,
):

    nsamp_start = int(np.log(nsamp_min) / np.log(nsamp_base))
    nsamp_end = int(np.ceil(np.log(nsamp_max) / np.log(nsamp_base)))
    nsamp_values = np.logspace(
        nsamp_start,
        nsamp_end,
        num=nsamp_end - nsamp_start + 1,
        base=nsamp_base,
        dtype=int,
    )

    np.random.seed(9434)
    option = AsianOption()

    means_2l = np.zeros(len(nsamp_values))
    variances_2l = np.zeros(len(nsamp_values))
    means_l0 = np.zeros(len(nsamp_values))
    variances_l0 = np.zeros(len(nsamp_values))
    means_l01 = np.zeros(len(nsamp_values))
    variances_l01 = np.zeros(len(nsamp_values))
    means_crude = np.zeros(len(nsamp_values))
    variances_crude = np.zeros(len(nsamp_values))
    optimal_ratios = np.zeros(len(nsamp_values))
    nsamp_crudes = np.zeros(len(nsamp_values))

    for i, nsamp0 in tqdm(enumerate(nsamp_values), total=len(nsamp_values)):

        # pilot run
        h_coarse = 0.2
        result_pilot = coarse_fine_mc(
            nsamp_pilot, h_coarse, option
        )

        # estimate optimal sample sizes
        optimal_ratio = np.sqrt(
            result_pilot["var_diff"] / (2 * result_pilot["var_coarse"])
        )
        optimal_ratios[i] = optimal_ratio
        nsamp1 = int(np.ceil(optimal_ratio * nsamp0))

        # run two-level Monte Carlo
        result_2l = two_level_mc(
            nsamp0, nsamp1, h_coarse, option
        )

        means_2l[i] = result_2l["esp"]
        variances_2l[i] = result_2l["var"]
        means_l0[i] = result_2l["esp0"]
        variances_l0[i] = result_2l["var0"]
        means_l01[i] = result_2l["esp01"]
        variances_l01[i] = result_2l["var01"]

        # run comparable crude Monte Carlo
        ncrude = int(np.ceil(nsamp0 * (0.5 + optimal_ratio)))
        nsamp_crudes[i] = ncrude
        result_crude = standard_mc(
            ncrude, 0.5 * h_coarse, option
        )

        means_crude[i] = result_crude["esp"]
        variances_crude[i] = result_crude["var"]

    df = pd.DataFrame(
        {
            "nsamp": nsamp_values,
            "optimal_ratio": optimal_ratios,
            "mean_2l": means_2l,
            "variance_2l": variances_2l,
            "mean_l0": means_l0,
            "variance_l0": variances_l0,
            "mean_l01": means_l01,
            "variance_l01": variances_l01,
            "nsamp_crude": nsamp_crudes,
            "mean_crude": means_crude,
            "variance_crude": variances_crude,
        }
    )

    out_path = out_dir / f"two_level_nsamp_pilot={nsamp_pilot}.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variance reduction achieved by the 2-level estimator."
    )
    parser.add_argument(
        "--nsamp_min", "-min", default=10, type=int, help="Minimum number of samples."
    )
    parser.add_argument(
        "--nsamp_max",
        "-max",
        default=100_000,
        type=int,
        help="Maximum number of samples.",
    )
    parser.add_argument(
        "--nsamp_base",
        "-b",
        default=10,
        type=int,
        help="Base for the number of samples.",
    )
    parser.add_argument("--nseeds", "-s", type=int, default=1, help="Number of seeds.")
    parser.add_argument(
        "--nsamp_pilot",
        "-p",
        default=1_000,
        type=int,
        help="Number of samples for the pilot run.",
    )
    args = parser.parse_args()
    main(
        args.nsamp_min,
        args.nsamp_max,
        args.nsamp_base,
        args.nsamp_pilot,
        DATA_DIR,
        args.nseeds,
    )
