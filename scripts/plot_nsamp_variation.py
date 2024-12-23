import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlmc.utils.plot import set_plot_style, NATURE, STYLES, LINEWIDTH_SIZE

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)


def main(nsteps_coarse: int, style: str = NATURE, usetex: bool = False) -> None:

    df = pd.read_csv(DATA_DIR / f"nsamp_variation_nsteps_coarse={nsteps_coarse}.csv")
    bias = df["bias"]
    var = df["variance_coarse"]
    var_diff = df["variance_diff"]

    bias_trend = bias.iloc[0] * np.sqrt(df["nsamp"].iloc[0]) / np.sqrt(df["nsamp"])
    var_trend = var.iloc[0] * df["nsamp"].iloc[0] / df["nsamp"]

    set_plot_style(style, usetex)
    fig_bias, ax_bias = plt.subplots(
        1, 1, figsize=LINEWIDTH_SIZE, layout="constrained", sharex=True
    )
    ax_bias.loglog(
        df["nsamp"],
        bias_trend,
        label=r"$\mathcal{O}(N^{-1/2})$",
        linestyle="--",
        color="black",
    )
    ax_bias.loglog(
        df["nsamp"],
        bias,
        label=r"$|\hat{\mu}_h - \hat{\mu}_{h/2}|$",
        marker="o",
    )
    # ax_bias.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_bias.set_xlabel("number of samples $N$")
    h, l = ax_bias.get_legend_handles_labels()
    h = h[1:] + [h[0]]
    l = l[1:] + [l[0]]
    ax_bias.legend(h, l, loc="best")

    fn = f"nsamp_variation-bias-nsteps_coarse={nsteps_coarse}.pdf"
    fig_bias.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")

    fig_var = plt.figure(figsize=LINEWIDTH_SIZE, layout="constrained")
    ax_var = fig_var.subplots(1, 1)
    ax_var.loglog(
        df["nsamp"],
        var_trend,
        label=r"$\mathcal{O}(N^{-1})$",
        linestyle="--",
        color="black",
    )
    ax_var.loglog(df["nsamp"], var, label=r"$\mathbb{V}[\hat{\mu}_h]$", marker="o")
    ax_var.loglog(
        df["nsamp"],
        var_diff,
        label=r"$\mathbb{V}[\hat{\mu}_h - \hat{\mu}_{h/2}]$",
        marker="o",
    )
    ax_var.set_ylim(None, 1e1)
    ax_var.set_xlabel("number of samples $N$")
    h, l = ax_var.get_legend_handles_labels()
    h = h[1:] + [h[0]]
    l = l[1:] + [l[0]]
    ax_var.legend(h, l, loc="best", ncols=3, fontsize=8)

    fn = f"nsamp_variation-variance-nsteps_coarse={nsteps_coarse}.pdf"
    fig_var.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsteps_coarse", type=int, default=1000, help="Number of coarse time steps"
    )
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsteps_coarse, args.style, args.usetex)
