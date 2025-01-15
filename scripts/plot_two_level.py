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

def main(
    nsamp_pilot: int, style: str = NATURE, usetex: bool = False
) -> None:

    df = pd.read_csv(
        DATA_DIR / f"two_level_nsamp_pilot={nsamp_pilot}.csv"
    )
    optimal_ratio = df["optimal_ratio"]
    mean_2l = df["mean_2l"]
    var_2l = df["variance_2l"]
    nsamp_crude = df["nsamp_crude"]
    mean_crude = df["mean_crude"]
    var_crude = df["variance_crude"]


    mean_ratio = np.mean(optimal_ratio)
    std_ratio = np.std(optimal_ratio)
    logger.info(f"Mean optimal ratio: {mean_ratio:.4f}±{std_ratio:.4f}")

    final_nsamp = df["nsamp"].iloc[-1]
    final_var_2l = var_2l.iloc[-1]
    final_var_crude = var_crude.iloc[-1]
    logger.info(f"Final number of coarse samples: {final_nsamp}")
    logger.info(f"Final variance (2-level): {final_var_2l:.3e}")
    logger.info(f"Final variance (crude): {final_var_crude:.3e}")


    set_plot_style(style, usetex)
    fig_conv = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_conv = fig_conv.subplots(1, 1)
    ax_conv.fill_between(df["nsamp"], mean_2l - np.sqrt(var_2l), mean_2l + np.sqrt(var_2l), alpha=0.3)
    ax_conv.loglog(df["nsamp"], mean_2l, label="$\hat{\mu}_h^{(2)}$", marker="o")

    ax_conv.fill_between(df["nsamp"], mean_crude - np.sqrt(var_crude), mean_crude + np.sqrt(var_crude), alpha=0.3)
    ax_conv.loglog(df["nsamp"], mean_crude, label="$\hat{\mu}_h^{\mathrm{MC}}$", marker="o")
    ax_conv.set_xlabel("coarse samples $N_0$")
    ax_conv.legend(loc="best")

    fn = PLOT_DIR / f"two_level-conv_nsamp_pilot={nsamp_pilot}.pdf"
    fig_conv.savefig(fn)
    print(f"Saved figure to {fn}")

    fig_var = plt.figure(figsize=(LINEWIDTH_SIZE[0] * 0.6, LINEWIDTH_SIZE[1]), constrained_layout=True)
    ax_var = fig_var.subplots(1, 1)
    ax_var.semilogx(df["nsamp"], 0.5*np.ones_like(df["nsamp"]), label="0.5", linestyle="--", color="black")
    ax_var.semilogx(df["nsamp"], var_2l / var_crude, marker="o", label="$\mathbb{V}[\hat{\mu}_h^{(2)}] / \mathbb{V}[\hat{\mu}_h^{\mathrm{MC}}]$")
    ax_var.set_ylim([0.3, 0.7])
    ax_var.set_xlabel("coarse samples $N_0$")
    h, l = ax_var.get_legend_handles_labels()
    ax_var.legend(h[::-1], l[::-1], loc="lower left", ncols=1)

    fn = PLOT_DIR / f"two_level-var_nsamp_pilot={nsamp_pilot}.pdf"
    fig_var.savefig(fn)
    print(f"Saved figure to {fn}")

    fig_ratio = plt.figure(figsize=(LINEWIDTH_SIZE[0] * 0.4, LINEWIDTH_SIZE[1]), constrained_layout=True)
    ax_ratio = fig_ratio.subplots(1, 1)
    ax_ratio.boxplot(optimal_ratio,
                    showmeans=True,
                    meanprops = {"markerfacecolor": "black", "markeredgecolor": "black"},
                    medianprops = {"color": "black"})
    ax_ratio.set_xticks([1])
    ax_ratio.set_xticklabels(["$N_1 / N_0$"])
    ax_ratio.set_xlabel("DUMMY", color="white")
    
    fn = PLOT_DIR / f"two_level-ratio_nsamp_pilot={nsamp_pilot}.pdf"
    fig_ratio.savefig(fn)
    print(f"Saved figure to {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsamp_pilot", type=int, default=1000, help="Number of coarse time steps"
    )
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsamp_pilot, args.style, args.usetex)
