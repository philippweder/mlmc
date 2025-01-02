import argparse
import logging
import numpy as np

from mlmc.core.estimators import standard_mc, antithetic_mc, ismc
from mlmc.core.options import AsianOption, BarrierOption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(nsamp: int, h: float) -> None:
    np.random.seed(9434)

    HIGH_STRIKE = 2.0
    HIGH_BARRIER = 2.5

    asian_option = AsianOption(K=HIGH_STRIKE)
    asian_mc = standard_mc(nsamp, h, asian_option)
    logger.info(
        f"Asian option | E[Y]: {asian_mc['esp']}, Var[Y]: {asian_mc['var']}"
    )
    asian_atmc = antithetic_mc(nsamp, h, asian_option)
    logger.info(
        f"Asian option (antithetic) | E[Y]: {asian_atmc['esp']}, Var[Y]: {asian_atmc['var']}"
    )

    R = 10 * asian_option.r
    asian_ismc = ismc(nsamp, h, asian_option, R)
    logger.info(
        f"Asian option (importance sampling) | E[Y]: {asian_ismc['esp']}, Var[Y]: {asian_ismc['var']}"
    )

    barrier_option = BarrierOption(K=HIGH_STRIKE, Smax=HIGH_BARRIER)
    barrier_mc = standard_mc(nsamp, h, barrier_option)
    logger.info(
        f"Barrier option | E[Y]: {barrier_mc['esp']}, Var[Y]: {barrier_mc['var']}"
    )
    barrier_atmc = antithetic_mc(nsamp, h, barrier_option)
    logger.info(
        f"Barrier option (antithetic) | E[Y]: {barrier_atmc['esp']}, Var[Y]: {barrier_atmc['var']}"
    )
    barrier_ismc = ismc(nsamp, h, barrier_option, R)
    logger.info(
        f"Barrier option (importance sampling) | E[Y]: {barrier_ismc['esp']}, Var[Y]: {barrier_ismc['var']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamp", type=int, default=1_000_000)
    parser.add_argument("--h", type=float, default=0.01)
    args = parser.parse_args()
    main(args.nsamp, args.h)
