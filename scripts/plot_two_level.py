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
    nsamp_pilot: int, nseeds: int = 1, style: str = NATURE, usetex: bool = False
) -> None:

    for nseed in range(nseeds):
        df = pd.read_csv(
            DATA_DIR / f"two_level_nsamp_pilot={nsamp_pilot}_seed={nseed}.csv"
        )
        if nseed == 0:
            optimal_ratio = df["optimal_ratio"]
            mean_2l = df["mean_2l"]
            var_2l = df["variance_2l"]
            nsamp_crude = df["nsamp_crude"]
            mean_crude = df["mean_crude"]
            var_crude = df["variance_crude"]

            V0_pilot = df["V0_pilot"]
            V1_pilot = df["V1_pilot"]
            V_crude = df["V_crude"]

        else:
            optimal_ratio += df["optimal_ratio"]
            mean_2l += df["mean_2l"]
            var_2l += df["variance_2l"]
            nsamp_crude += df["nsamp_crude"]
            mean_crude += df["mean_crude"]
            var_crude += df["variance_crude"]

            V0_pilot += df["V0_pilot"]
            V1_pilot += df["V1_pilot"]
            V_crude += df["V_crude"]

    optimal_ratio /= nseeds
    mean_2l /= nseeds
    var_2l /= nseeds
    nsamp_crude /= nseeds
    mean_crude /= nseeds
    var_crude /= nseeds

    V0_pilot /= nseeds
    V1_pilot /= nseeds
    V_crude /= nseeds

    set_plot_style(style, usetex)
    fig, ((ax_eff, ax_nsamp), (ax_cost, ax_mean)) = plt.subplots(
        2, 2, figsize=(2 * LINEWIDTH_SIZE[0], 4), layout="constrained", sharex=True
    )

    ax_eff.loglog(
        df["nsamp"],
        var_2l / var_crude,
        label="$\mathrm{V}[\hat{\mu}_{h}^{(2)}] / \mathrm{V}[\hat{\mu}_{h/2}^{\mathrm{crude}}] $",
        marker="o",
    )

    ax_eff.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_eff.legend(loc="best")

    ax_nsamp.loglog(df["nsamp"], df["nsamp"], label="$N_0$", ls="--", color="black")
    ax_nsamp.loglog(df["nsamp"], nsamp_crude, label="$N_c$", marker="o")
    # ax_nsamp.loglog(df["nsamp"], optimal_ratio * df["nsamp"], label="$N_1$", marker="s")
    # ax_nsamp.loglog(df["nsamp"], (1 + optimal_ratio) * df["nsamp"], label="$N_0 + N_1$", marker="d")
    
    ax_nsamp.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    # ax_nsamp.set_xlabel("number of coarse samples $N_0$")
    ax_nsamp.legend(loc="best")

    ax_cost.loglog(df["nsamp"], np.ones_like(df["nsamp"]), label="$1$", ls="--", color="black")
    ax_cost.loglog(df["nsamp"], optimal_ratio, label="optimal ratio", marker="o")
    ax_cost.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_cost.set_xlabel("number of coarse samples $N_0$")
    ax_cost.legend(loc="best")

    ax_mean.loglog(df["nsamp"], mean_2l, label="$\hat{\mu}_h^{(2)}$", marker="o")
    ax_mean.loglog(df["nsamp"], mean_crude, label="$\hat{\mu}_{h/2}^{\mathrm{crude}}$", marker="s")
    ax_mean.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_mean.set_xlabel("number of coarse samples $N_0$")
    ax_mean.legend(loc="best")

    fn = f"two-level_nsamp_pilot={nsamp_pilot}_nseeds={nseeds}.pdf"
    fig.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")

    fig = plt.figure(layout="constrained", figsize=(LINEWIDTH_SIZE[0], 4))
    ax_ratio, ax_pilot = fig.subplots(2, 1, sharex=True)
    ax_ratio.semilogy(df["nsamp"], V0_pilot / V_crude, label="$V_0 / \\tilde{V}_0$", marker="o")
    ax_ratio.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_ratio.legend(loc="best")
    
    ax_pilot.loglog(df["nsamp"], V0_pilot, label="$V_0$", marker="o")
    ax_pilot.loglog(df["nsamp"], V1_pilot, label="$V_1$", marker="s")
    ax_pilot.set_xlim(df["nsamp"].min(), df["nsamp"].max())
    ax_pilot.legend(loc="best")
    ax_pilot.set_xlabel("number of coarse samples $N_0$")

    fn = f"two-level_diganostics_pilot={nsamp_pilot}_nseeds={nseeds}.pdf"
    fig.savefig(PLOT_DIR / fn)
    print(f"Plot saved to {PLOT_DIR / fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to analyze the variation of the number of samples."
    )
    parser.add_argument(
        "--nsamp_pilot", type=int, default=1000, help="Number of coarse time steps"
    )
    parser.add_argument("--nseeds", type=int, default=1, help="Number of seeds")
    parser.add_argument(
        "--style", type=str, default=NATURE, choices=STYLES, help="Plot style"
    )
    parser.add_argument(
        "--usetex", action="store_true", help="Use LaTeX for text rendering"
    )
    args = parser.parse_args()

    main(args.nsamp_pilot, args.nseeds, args.style, args.usetex)
