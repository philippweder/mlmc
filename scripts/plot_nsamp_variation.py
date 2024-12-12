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
    
    df = pd.read_csv(DATA_DIR / f"nsamp_variation-nsteps_coarse={nsteps_coarse}.csv")

    bias_trend = df["bias"].iloc[0] * np.sqrt(df["nsamp"].iloc[0]) / np.sqrt(df["nsamp"])

    set_plot_style(style, usetex)
    fig, ax = plt.subplots(figsize=LINEWIDTH_SIZE, layout="constrained")
    ax.loglog(df["nsamp"], bias_trend, label="$\mathcal{O}(N^{-1/2})$", linestyle="--", color="black")
    ax.loglog(df["nsamp"], df["bias"], label="bias", marker="o")
    ax.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax.set_xlabel("Number of samples $N$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    
    fn = f"nsamp_variation-nsteps_coarse={nsteps_coarse}.pdf"
    fig.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsteps_coarse", type=int, default=100, help="Number of coarse time steps"
    )
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument("--usetex", action="store_true", help="Use LaTeX for text rendering")
    args = parser.parse_args()

    main(args.nsteps_coarse, args.style, args.usetex)

