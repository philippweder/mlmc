import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlmc.utils.plot import set_plot_style, NATURE, STYLES, LINEWIDTH_SIZE

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)


def main(nsamp: int, style: str = NATURE, usetex: bool = False) -> None:

    df = pd.read_csv(DATA_DIR / f"h_variation-nsamp={nsamp}.csv")

    set_plot_style(style, usetex)
    fig, ax = plt.subplots(figsize=LINEWIDTH_SIZE, layout="constrained")

    variance_diff_trend = df["variance_diff"].iloc[0] * df["h"].iloc[0] * df["h"]
    ax.loglog(
        df["h"],
        variance_diff_trend,
        label="$\mathcal{O}(h)$",
        linestyle="--",
        color="black",
    )
    ax.loglog(df["h"], df["variance_diff"], label="variance diff", marker="o")
    ax.set_xlim(df["h"].min(), df["h"].max())
    ax.set_xlabel("Time step size $h$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="best")

    fn = f"h_variation-nsamp={nsamp}.pdf"
    fig.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the time step."
    )
    parser.add_argument("--nsamp", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsamp, args.style, args.usetex)
