import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from mlmc.utils.plot import set_plot_style, NATURE, STYLES, LINEWIDTH_SIZE

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(nsamp_pilot: int, nlevels_pilot: int, usetex: bool = False) -> None:
    fp = DATA_DIR / f"mlmc-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.csv"
    df = pd.read_csv(fp)

    fp_levels = (
        DATA_DIR
        / f"mlmc_nlevels-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.csv"
    )
    df_levels = pd.read_csv(fp_levels)

    set_plot_style(usetex=usetex)
    fig_cost = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_cost = fig_cost.subplots(1, 1)
    ax_cost.loglog(df["eps"], df["cpu_time"], marker="o", label="MLMC")
    ax_cost.set_xlabel(r"$\varepsilon$")
    ax_cost.set_ylabel("CPU time (s)")
    ax_cost.legend(loc="best")

    fn = (
        PLOT_DIR
        / f"mlmc_cost-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_cost.savefig(fn)
    logger.info(f"Plot saved to {fn}")

    fig_levels = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_levels = fig_levels.subplots(1, 1)
    for i, row in df_levels.iterrows():
        levels = [int(row[c]) for c in row.index if c != "eps" and not row[c] == 0]
        levels = np.array(levels)
        ax_levels.semilogy(np.arange(len(levels)), levels, marker="o", label = f"$\\varepsilon = {row['eps']:0e}$")

    ax_levels.set_xlabel("Level $\ell$")
    ax_levels.set_ylabel("Number of samples $N_{\ell}$")
    ax_levels.legend(loc="best")

    fn = (
        PLOT_DIR
        / f"mlmc_levels-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_levels.savefig(fn)
    logger.info(f"Plot saved to {fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to plot the results of the MLMC simulation."
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
        "--usetex",
        action="store_true",
        help="Use LaTeX for the plots.",
    )

    args = parser.parse_args()
    main(args.nsamp_pilot, args.nlevels_pilot, args.usetex)
