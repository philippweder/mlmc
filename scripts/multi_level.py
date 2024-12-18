import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time 
import warnings

from mlmc.core.estimators import mlmc
from mlmc.core.helpers import mlmc_pilot
from mlmc.core.payoff import asian_option

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_DIR = Path("../data/asian_option/mlmc_runs")
DATA_DIR.mkdir(exist_ok=True, parents=True)

payoff_params = {
    "T": 1,  # Time to maturity
    "r": 0.05,  # Risk-free interest rate
    "sigma": 0.2,  # Volatility
    "K": 1,  # Strike price
    "S0": 1,  # Initial stock price
}


def main(
    eps_min: float, #we will try to achieve MSE< 2*epsilon . The variables eps stand for this epsilon
    eps_max: float,
    eps_base: float,
    need_pilot_run: bool, #do you need the pilot run or do you already have E0 and V0
    nsamp_pilot: int,
    nlevels_pilot: int,
    out_dir: Path,
):
    #np.random.seed(9434)
    np.random.seed(9336)
    h_coarse = 0.2 #this value for h0 is forced by the statement of the project. 
    eps_start = int(np.log(eps_min) / np.log(eps_base))
    eps_end = int(np.floor(np.log(eps_max) / np.log(eps_base)))
    eps_values = np.logspace(
        eps_start,
        eps_end,
        num=2*(eps_end - eps_start + 1),
        base=eps_base,
    ) #creates logaritmically (base eps_base) spaced values for epsilon, ranging from eps_min to eps_max
    
    
    means = np.zeros(len(eps_values))
    variances = np.zeros(len(eps_values))
    nlevels = np.zeros(len(eps_values)) # L 
    cpu_time = np.zeros(len(eps_values))
    N0 = np.zeros(len(eps_values))
    
    if need_pilot_run:
        if nsamp_pilot==int(50000) and nlevels_pilot==int(8):
                warnings.warn("For these values of nsamp:pilot and nlevels_pilot," 
                    +"E0 and V0 are already computed you can set need_pilot_run = 0",
                    category=Warning)
        pilot_results = mlmc_pilot(
             nlevels_pilot, nsamp_pilot, h_coarse, asian_option, **payoff_params)
        E0= pilot_results["E0"],
        V0 = pilot_results["V0"],
    else:
        if nsamp_pilot!=int(50000) and nlevels_pilot!=int(8):
                warnings.warn("E0 and V0 used in this run were computed using different values" 
                    +"of nsamp and nlevels for the pilot run than the ones you provided." +
                    "If you want to recompute it set parameter need_pilot_run = 1.",
                    category=Warning)
        E0 = 1.49e-4
        V0 = 2.56e-5
        
    pbar = tqdm(
        enumerate(eps_values), total=len(eps_values), desc="Scanning epsilon values"
    )

    for i, eps in pbar:
        print(f"epsilon = {eps}")
        start_cpu_time = time.process_time()
        
        optimal_nlevels = int(np.ceil(np.log2(E0 / eps)))
        optimal_nsamps = [
            int(np.ceil(2 **(-2*l) * (optimal_nlevels + 1) * V0 / eps**2))
            for l in range(optimal_nlevels)
        ]
        
        logger.info(f"Optimal number of levels: {optimal_nlevels}")
        logger.info(f"Optimal number of samples: {optimal_nsamps}")
        
        print(f"Optimal number of levels: {optimal_nlevels}")
        print(f"Optimal number of samples: {optimal_nsamps}")

        #below : run the actual MLMC simulation using optimal Nl and L
        result = mlmc(optimal_nsamps, h_coarse, asian_option, **payoff_params)
        end_cpu_time = time.process_time()
        
        
        cpu_time[i] = end_cpu_time - start_cpu_time
        means[i] = result["esp"]
        variances[i] = result["var"] #this is var(Yl - Yl-1)/nsamp
        nlevels[i] = optimal_nlevels
        N0[i] = optimal_nsamps[0] #number of samples needed at first level l=0.
        
        print(f"CPU Time: {cpu_time[i]:.6f} seconds")

    df = pd.DataFrame(
        {
            "eps": eps_values,
            "cpu_time": cpu_time,
            "mean": means, #this is for each level esp(Yl - Yl-1)
            "variance": variances, #this is var(Yl - Yl-1)/nsamp
            "nlevels": nlevels,
            "N0": N0, #number of samples needed at first level l=0
        }
    )

    out_path = (
        out_dir
        / f"mlmc_epsmax={eps_max}_epsmin={eps_min}_nsamp_pilot={nsamp_pilot}_nlevels_pilot={nlevels_pilot}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

argParse = 0 #set to zero if you run the file from an IDE, to 1 to run from command line 


if argParse:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Script to analyze the variation of the MSE for the MLMC estimator."
        )
        parser.add_argument(
            "--eps_min", type=float, default=1e-10, help="Minimum eps value, where we want to achieve MSE<2 eps^2."
        )
        parser.add_argument(
            "--eps_max", type=float, default=1e-6, help="Maximum MSE value, where we want to achieve MSE<2 eps^2."
        )
        parser.add_argument(
            "--eps_base", type=float, default=10, help="Base for the eps values."
        )
        
        parser.add_argument(
            "--need_pilot_run", type=bool, default=0, help="Do you want to run pilot run or to use already estimated"+
            "values for E0 and V0"
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
            default=8,
            help="Number of levels for the pilot run.",
        )
        args = parser.parse_args()
    
        main(
            args.eps_min,
            args.eps_max,
            args.eps_base,
            args.need_pilot_run,
            args.nsamp_pilot,
            args.nlevels_pilot,
            DATA_DIR,
        )
else:
    if __name__ == "__main__":

        main(
            eps_min=1e-6,
            eps_max=1e-4,
            eps_base=10,
            need_pilot_run=0, 
            nsamp_pilot=50000,
            nlevels_pilot=8,
            out_dir=DATA_DIR,
        )
        
        
      
