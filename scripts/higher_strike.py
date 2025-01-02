import argparse
import logging
import numpy as np

from mlmc.core.estimators import standard_mc, antithetic_mc
from mlmc.core.options import AsianOption, BarrierOption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(nsamp: int, h: float) -> None:
    np.random.seed(9434)

    HIGH_STRIKE = 2.0
    HIGH_BARRIER = 2.5

    asian_option = AsianOption(K=HIGH_STRIKE)
    asian_result = standard_mc(nsamp, h, asian_option)
    logger.info(
        f"Asian option | E[Y]: {asian_result['esp']}, Var[Y]: {asian_result['var']}"
    )
    asian_result_antithetic = antithetic_mc(nsamp, h, asian_option)
    logger.info(
        f"Asian option (antithetic) | E[Y]: {asian_result_antithetic['esp']}, Var[Y]: {asian_result_antithetic['var']}"
    )

    barrier_option = BarrierOption(K=HIGH_STRIKE, Smax=HIGH_BARRIER)
    barrier_result = standard_mc(nsamp, h, barrier_option)
    logger.info(
        f"Barrier option | E[Y]: {barrier_result['esp']}, Var[Y]: {barrier_result['var']}"
    )
    barrier_result_antithetic = antithetic_mc(nsamp, h, barrier_option)
    logger.info(
        f"Barrier option (antithetic) | E[Y]: {barrier_result_antithetic['esp']}, Var[Y]: {barrier_result_antithetic['var']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamp", type=int, default=50_000)
    parser.add_argument("--h", type=float, default=0.1)
    args = parser.parse_args()
    main(args.nsamp, args.h)
