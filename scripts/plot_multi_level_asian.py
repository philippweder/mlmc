import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from mlmc.utils.plot import set_plot_style, NATURE, LINEWIDTH_SIZE, MARKERS, format_scientific

DATA_DIR = Path("../data/asian_option")
PLOT_DIR = Path("../plots/asian_option")
PLOT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(nsamp_pilot: int, nlevels_pilot: int, coeffs: str="estimated", usetex: bool = False) -> None:
    fp = DATA_DIR / f"mlmc-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={coeffs}.csv"
    df = pd.read_csv(fp)

    fp_levels = (
        DATA_DIR
        / f"mlmc_nlevels-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={coeffs}.csv"
    )
    df_levels = pd.read_csv(fp_levels)

    fp_pilot = (
        DATA_DIR
        / f"mlmc_pilot_nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}_coeffs={coeffs}.csv"
    )
    df_pilot = pd.read_csv(fp_pilot)

    set_plot_style(usetex=usetex, color_style=NATURE)
    fig_pilot = plt.figure(
        figsize=(LINEWIDTH_SIZE[0], LINEWIDTH_SIZE[1]), constrained_layout=True
    )
    (ax_bias, ax_variance) = fig_pilot.subplots(1, 2, sharex=True)

    levels = np.arange(1, nlevels_pilot + 1)
    E0 = df["E0"].iloc[0]
    alpha = df["alpha"].iloc[0]
    V0 = df["V0"].iloc[0]
    beta = df["beta"].iloc[0]

    logger.info(f"E0 = {E0:.3e}, alpha = {alpha:.3e}, V0 = {V0:.3e}, beta = {beta:.3e}")

    ax_bias.loglog(
        levels,
        E0 / (2 ** (alpha * levels)),
        label="$\\Tilde{E}_0 2^{-\\alpha\ell}$",
        ls="--",
        color="black",
    )
    ax_bias.scatter(
        levels, df_pilot["biases"], marker="o", label="$E_\ell$", linewidths=0.5
    )
    ax_bias.set_xlabel("level $\ell$")
    ax_bias.set_ylabel("DUMMY", color="white")
    ax_bias.legend(loc="best")

    ax_variance.loglog(
        levels,
        V0 / (2 ** (beta * levels)),
        label="$\\Tilde{V}_0 2^{-\\beta\ell}$",
        ls="--",
        color="black",
    )
    ax_variance.scatter(
        levels, df_pilot["variances"], marker="o", label="$V_\ell$", linewidths=0.5
    )
    ax_variance.set_xlabel("level $\ell$")
    ax_variance.set_yscale("log")
    ax_variance.set_xscale("linear")
    ax_variance.legend(loc="best")

    fn = (
        PLOT_DIR
        / f"mlmc_pilot-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_pilot.savefig(fn)
    logger.info(f"Plot saved to {fn}")

    fig_cpu = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_cpu = fig_cpu.subplots(1, 1)
    ax_cpu.loglog(df["eps"], df["cpu_time_mlmc"], marker="o", label="MLMC")
    ax_cpu.loglog(df["eps"], df["cpu_time_mc"], marker="s", label="MC")
    ax_cpu.set_xlabel(r"target precision $\varepsilon$")
    ax_cpu.set_ylabel("CPU time [s]")
    ax_cpu.legend(loc="best")

    fn = (
        PLOT_DIR
        / f"mlmc_cpu-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_cpu.savefig(fn)
    logger.info(f"Plot saved to {fn}")

    fig_levels = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_levels = fig_levels.subplots(1, 1)
    for (i, row), marker in zip(df_levels.iterrows(), MARKERS):
        nsamps = [int(row[c]) for c in row.index if c != "eps" and not row[c] == 0]
        nsamps = np.array(nsamps)
        eps = row["eps"]
        eps_str = format_scientific(eps, 0)
        ax_levels.semilogy(
            np.arange(len(nsamps)),
            nsamps,
            marker=marker,
            label=f"${eps_str}$",
        )

    ax_levels.set_xlabel("level $\ell$")
    ax_levels.set_ylabel("number of samples $N_{\ell}$")
    ax_levels.legend(loc="lower right", ncols=2, title="target precision $\\varepsilon$")
    ax_levels.set_ylim(1e-2, 1e8)

    fn = (
        PLOT_DIR
        / f"mlmc_levels-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_levels.savefig(fn)
    logger.info(f"Plot saved to {fn}")

    fig_cost = plt.figure(figsize=LINEWIDTH_SIZE, constrained_layout=True)
    ax_cost = fig_cost.subplots(1, 1)

    mlmc_cost = []
    for _, row in df_levels.iterrows():
        nsamps = [int(row[c]) for c in row.index if c != "eps" and not row[c] == 0]
        nsamps = np.array(nsamps)
        cost = nsamps[0]
        for l, nsamp in enumerate(nsamps[1:]):
            cost += nsamp * (2 ** (l + 1) + 2**l)
        mlmc_cost.append(cost)
    mlmc_trend = (
    df["eps"] ** (-2)
    * np.log(df["eps"]) ** 2
    * mlmc_cost[0]
    * (df["eps"].iloc[0] ** 2)
    / (np.log(df["eps"].iloc[0])) ** 2
    )

    mc_cost = df["nsamp_mc"] * (2 ** df["nlevels"])
    mc_trend = df["eps"] ** (-3) * mc_cost.iloc[-1] * (df["eps"].iloc[-1] ** 3)

    ax_cost.loglog(
        df["eps"],
        mlmc_trend,
        label="$\mathcal{O}(\\varepsilon^{-2} |\log(\\varepsilon)|^2)$",
        color="black",
        ls="--",
    )
    ax_cost.loglog(
        df["eps"],
        mc_trend,
        label="$\mathcal{O}(\\varepsilon^{-3})$",
        color="black",
        ls="-.",
    )
    ax_cost.loglog(df["eps"], mlmc_cost, marker="o", label=f"MLMC")
    ax_cost.loglog(df["eps"], mc_cost, marker="s", label=f"MC")
    ax_cost.set_xlabel("target precision $\\varepsilon$")
    ax_cost.set_ylabel("normalized cost")
    ax_cost.set_ylim(1e1, 1e12)

    h, l = ax_cost.get_legend_handles_labels()
    h = [h[2], h[3], h[0], h[1]]
    l = [l[2], l[3], l[0], l[1]]
    ax_cost.legend(h, l, loc="best", ncols=2)

    fn = (
        PLOT_DIR
        / f"mlmc_cost-nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.pdf"
    )
    fig_cost.savefig(fn)
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
        default=6,
        help="Number of levels for the pilot run.",
    )

    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use LaTeX for the plots.",
    )
    parser.add_argument(
        "--coeffs",
        "-c",
        type=str,
        default="prescribed",
        choices=["estimated", "prescribed"],
        help="""Whether to use the results from estimating
            alpha and beta or the results from a fixed value""",
    )

    args = parser.parse_args()
    main(args.nsamp_pilot, args.nlevels_pilot, args.coeffs, args.usetex)

