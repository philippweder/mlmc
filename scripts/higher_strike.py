import argparse
import logging
import numpy as np

from mlmc.core.estimators import standard_mc, ic_mc
from mlmc.core.options import AsianOption, BarrierOption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(nsamp: int, h: float, payoff: str = "both") -> None:
    np.random.seed(9434)

    HIGH_STRIKE = 2
    HIGH_BARRIER = 2.5

    if payoff == "asian_option" or payoff == "both":

        asian_option = AsianOption(K=HIGH_STRIKE)
        asian_mc = standard_mc(nsamp, h, asian_option)
        logger.info(
            f"Asian option | E[Y]: {asian_mc['esp']}, Var[Y]: {asian_mc['var']}"
        )

        R = 10 * asian_option.r  # higher interest rate for dominating distr.

        asian_is = ic_mc(nsamp, h, asian_option, R)
        logger.info(
            f"Asian option (importance sampling changing drift in b.m.) | E[Y]: {asian_is['esp']}, Var[Y]: {asian_is['var']}"
        )

    if payoff == "barrier_option" or payoff == "both":
        barrier_option = BarrierOption(K=HIGH_STRIKE, Smax=HIGH_BARRIER)
        barrier_mc = standard_mc(nsamp, h, barrier_option)
        logger.info(
            f"Barrier option | E[Y]: {barrier_mc['esp']}, Var[Y]: {barrier_mc['var']}"
        )

        R = 10 * barrier_option.r  # higher interest rate for dominating distr.

        barrier_is = ic_mc(nsamp, h, barrier_option, R)
        logger.info(
            f"Barrier option (importance sampling changing drift in b.m.) | E[Y]: {barrier_is['esp']}, Var[Y]: {barrier_is['var']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamp", type=int, default=1_00_000)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--payoff", type=str, default="both")  # which payoff to use
    args = parser.parse_args()
    main(args.nsamp, args.h, args.payoff)
