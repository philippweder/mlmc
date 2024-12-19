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
    nsamp: int, nseeds: int = 1, style: str = NATURE, usetex: bool = False
) -> None:

    for nseed in range(nseeds):
        df = pd.read_csv(DATA_DIR / f"h_variation_nsamp={nsamp}_seed={nseed}.csv")
        if nseed == 0:
            bias = df["bias"]
            var = df["variance_coarse"]
        else:
            bias += df["bias"]
            var += df["variance_coarse"]

    bias /= nseeds
    bias_trend_one = bias.iloc[0] * df["h"] / df["h"].iloc[0]

    var /= nseeds
    var_trend = var.iloc[-1] * np.ones_like(df["h"])

    set_plot_style(style, usetex)
    fig_bias, ax_bias = plt.subplots(
        1, 1, figsize=LINEWIDTH_SIZE, layout="constrained", sharex=True
    )
    ax_bias.loglog(
        df["h"],
        bias_trend_one,
        label="$\mathcal{O}(h)$",
        linestyle="--",
        color="black",
    )
    ax_bias.loglog(
        df["h"],
        bias,
        label=r"$|\hat{\mu}_h - \hat{\mu}_{h/2}|$",
        marker="o",
    )
    h, l = ax_bias.get_legend_handles_labels()
    h = [h[1], h[0]]
    l = [l[1], l[0]]
    ax_bias.legend(h, l, loc="best")
    ax_bias.set_xlabel("time step size $h$")

    fn = f"h_variation-bias-nsamp={nsamp}_nseeds={nseeds}.pdf"
    fig_bias.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")

    fig_var, ax_var = plt.subplots(
        1, 1, figsize=LINEWIDTH_SIZE, layout="constrained", sharex=True
    )

    ax_var.loglog(
        df["h"],
        var_trend,
        label=r"$\mathcal{O}(1)$",
        linestyle="--",
        color="black",
    )
    ax_var.loglog(df["h"], var, label=r"$\mathbb{V}[Y_h]$", marker="o")
    ax_var.set_xlabel("time step size $h$")
    ax_var.set_ylim(1e-7, 1e-5)
    h, l = ax_var.get_legend_handles_labels()
    h = [h[1], h[0]]
    l = [l[1], l[0]]
    ax_var.legend(h, l, loc="best")

    fn = f"h_variation-variance-nsamp={nsamp}_nseeds={nseeds}.pdf"
    fig_var.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the time step."
    )
    parser.add_argument("--nsamp", type=int, default=10000, help="Number of samples")
    parser.add_argument("--nseeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsamp, args.nseeds, args.style, args.usetex)
