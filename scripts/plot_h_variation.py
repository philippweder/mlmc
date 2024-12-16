import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlmc.utils.plot import set_plot_style, NATURE, STYLES, LINEWIDTH_SIZE

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)


def main(nsamp: int, nseeds: int = 1, style: str = NATURE, usetex: bool = False) -> None:

    for nseed in range(nseeds):
        df = pd.read_csv(
            DATA_DIR / f"h_variation_nsamp={nsamp}_seed={nseed}.csv"
        )
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
    fig, (ax_bias, ax_var) = plt.subplots(
        2, 1, figsize=(LINEWIDTH_SIZE[0], 4), layout="constrained", sharex=True
    )
    ax_bias.loglog(
        df["h"],
        bias_trend_one,
        label="$\mathcal{O}(h)$",
        linestyle="--",
        color="black",
    )
    ax_bias.loglog(df["h"], bias, label=r"$\mathrm{E}[Y_h - Y_{h/2}]$", marker="o")
    ax_bias.set_xlim(df["h"].min(), df["h"].max())
    ax_bias.legend(loc="best")

    ax_var.loglog(
        df["h"],
        var_trend,
        label=r"$\mathcal{O}(1)$",
        linestyle="--",
        color="black",
    )
    ax_var.loglog(df["h"], var, label=r"$\mathrm{V}[Y_h]$", marker="o")
    ax_var.set_xlim(df["h"].min(), df["h"].max())
    ax_var.set_xlabel("time step size $h$")
    ax_var.legend(loc="best")

    fn = f"h_variation-nsamp={nsamp}_nseeds={nseeds}.pdf"
    fig.savefig(PLOT_DIR / fn)
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
    
