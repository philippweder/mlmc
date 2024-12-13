import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlmc.utils.plot import set_plot_style, NATURE, STYLES, LINEWIDTH_SIZE

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)


def main(
    nsteps_coarse: int, nseeds: int = 1, style: str = NATURE, usetex: bool = False
) -> None:

    for nseed in range(nseeds):
        df = pd.read_csv(
            DATA_DIR / f"nsamp_variation_nsteps_coarse={nsteps_coarse}_seed={nseed}.csv"
        )
        if nseed == 0:
            bias = df["bias"]
            var = df["variance_coarse"]
        else:
            bias += df["bias"]
            var += df["variance_coarse"]

    bias /= nseeds
    bias_trend = bias.iloc[0] * np.sqrt(df["nsamp"].iloc[0]) / np.sqrt(df["nsamp"])

    var /= nseeds
    var_trend = var.iloc[0] * df["nsamp"].iloc[0] / df["nsamp"]

    set_plot_style(style, usetex)
    fig, (ax_bias, ax_var) = plt.subplots(
        2, 1, figsize=(LINEWIDTH_SIZE[0], 4), layout="constrained", sharex=True
    )
    ax_bias.loglog(
        df["nsamp"],
        bias_trend,
        label="$\mathcal{O}(N^{-1/2})$",
        linestyle="--",
        color="black",
    )
    ax_bias.loglog(df["nsamp"], df["bias"], label=r"$\mathrm{E}[Y_h - Y_{h/2}]$", marker="o")
    ax_bias.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_bias.legend(loc="best")

    ax_var.loglog(
        df["nsamp"],
        var_trend,
        label="$\mathcal{O}(N^{-1})$",
        linestyle="--",
        color="black",
    )
    ax_var.loglog(df["nsamp"], var, label="$\mathrm{V}[Y_h]$", marker="o")
    ax_var.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_var.set_xlabel("number of samples $N$")
    ax_var.legend(loc="best")

    fn = f"nsamp_variation-nsteps_coarse={nsteps_coarse}_nseeds={nseeds}.pdf"
    fig.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsteps_coarse", type=int, default=100, help="Number of coarse time steps"
    )
    parser.add_argument("--nseeds", type=int, default=1, help="Number of seeds")
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsteps_coarse, args.nseeds, args.style, args.usetex)
