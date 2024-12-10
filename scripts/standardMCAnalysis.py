# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:39:12 2024

@author: mcost
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import the tqdm package for progress bar functionality

# Define constants for the problem
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
S0 = 1  # Initial stock price
K = 1  # Strike price
T = 1  # Time to maturity


#function standard MC on N realisations, for time step h (which will be compared 
#to time step h/2 for the bias):
def StandardMC(S0, r, sigma, K, T, N, h_coarse):
    
    #define the finer grid to estimate the bias:
    h_fine = h_coarse /2 
    M_coarse = int(T/h_coarse)
    M_fine = int(T/h_fine)

    # Pre-allocate storage for the payoffs
    payoffs_coarse = np.zeros(N)  # Array to store the payoffs for each path, coarse time grid
    payoffs_fine = np.zeros(N)  # Array to store the payoffs for each path, finer time grid

    # Simulate N independent paths of the geometric Brownian motion
    for n in tqdm(range(N), desc="Simulating Paths"):  # Add progress bar for the loop
        S_fine = np.zeros(M_fine + 1)  # Array to store the asset prices for a single path
        S_fine[0] = S0  # Initial stock price
        S_coarse = np.zeros(M_coarse + 1)
        S_coarse[0] = S0
    
        # Obtain the Brownian increments on the fine grid
        dW_fine = np.random.normal(0, np.sqrt(h_fine), size=M_fine)
        
        # Simulate the path using the Euler-Maruyama scheme, first on the finer grid
        for m in range(M_fine):
            # Update stock price using Euler-Maruyama scheme
            S_fine[m + 1] = S_fine[m] + r * S_fine[m] * h_fine + sigma * S_fine[m] * dW_fine[m]
        
        # Sum the Brownian increments to create coarser grid increments
        dW_coarse = np.add.reduceat(dW_fine, np.arange(0, M_fine, 2))
            
        # Simulate the path now on the coarse grid
        for m in range(M_coarse):
            S_coarse[m + 1] = S_coarse[m] + r * S_coarse[m] * h_coarse + sigma * S_coarse[m] * dW_coarse[m]
        
        # Approximate the arithmetic average price, for fine and coarse grids
        Sbar_fine = h_fine * np.sum((S_fine[:-1] + S_fine[1:]) / 2)
        Sbar_coarse = h_coarse * np.sum((S_coarse[:-1] + S_coarse[1:]) / 2)
    
        # Compute the discounted payoff for the Asian option
        payoffs_fine[n] = np.exp(-r) * max(0, Sbar_fine - K)
        payoffs_coarse[n] = np.exp(-r) * max(0, Sbar_coarse - K)
    
    result = {
        'esp_coarse': np.mean(payoffs_coarse),
        'esp_fine': np.mean(payoffs_fine),
        'var_coarse':np.var(payoffs_coarse, ddof = 1)/N,
        'var_fine' : np.var(payoffs_fine, ddof = 1)/N,
        # quantity below would be E[Y_L] in the statement
        'esp_diff' : np.mean(payoffs_fine - payoffs_coarse),
        'var_diff' : np.var(payoffs_fine - payoffs_coarse, ddof = 1),
        'bias': abs(np.mean(payoffs_coarse) - np.mean(payoffs_fine))
    }
    
    return result

#function to look at what happens when we vary N, number of MC realisations:
def VaryN_MC(S0, r, sigma, K, T, h_coarse, Nstart, Nend, Nit):
    
    Nvec = np.linspace(Nstart, Nend, Nit, dtype=int)
    bias = np.zeros(Nit)  
    esp_coarse = np.zeros(Nit)
    var_coarse = np.zeros(Nit)
    
    for j in range(Nit):
       result = StandardMC(S0, r, sigma, K, T, Nvec[j], h_coarse)
       bias[j] = result['bias']
       esp_coarse[j] = result['esp_coarse']
       var_coarse[j] = result['var_coarse']
     
    resultScanN = {
        'Nvec' : Nvec,
        'esp_coarse_estimator': esp_coarse,
        'var_coarse_estimator': var_coarse,
        'bias': bias
    }
    return resultScanN

#function to look at what happens when we vary h, the time step.

def Vary_h_coarse(S0, r, sigma, K, T,  N, hstart, hend, Nit):
    
    hvec = np.linspace(hstart, hend, Nit)
    bias = np.zeros(Nit)  
    esp_coarse = np.zeros(Nit)
    var_coarse = np.zeros(Nit)
    esp_diff = np.zeros(Nit)
    var_diff = np.zeros(Nit)
    
    for j in range(Nit):
       result = StandardMC(S0, r, sigma, K, T, N, hvec[j])
       bias[j] = result['bias']
       esp_coarse[j] = result['esp_coarse']
       var_coarse[j] = result['var_coarse']
       esp_diff[j] = result['esp_diff']
       var_diff[j] = result['var_diff']
       
    resultScanh = {
        'hvec' : hvec,
        'esp_coarse_estimator': esp_coarse,
        'var_coarse_estimator': var_coarse,
        'bias': bias,
        'esp_diff': esp_diff,
        'var_diff': var_diff
    }
    return resultScanh
    
    
####     if you want to do a single run of standard MC :
 
singleRunMC = 0
if singleRunMC:
    M_coarse = 1000  # Number of time steps (1/h)
    #Warning : we are also going to simulate 2M, to compare with finer grid. 
    N = 100  # Number of simulated paths
    h_coarse = T / M_coarse  # Step size for discretization
    resultMC = StandardMC(S0, r, sigma, K, T, N, h_coarse)
    # Print results
    print(f"Estimated Asian Option Price using coarse grid: {resultMC['esp_coarse']:.8f}")
    print(f"Estimated Asian Option Price using finer grid: {resultMC['esp_fine']:.8f}")
    print(f"Estimated bias: {resultMC['bias']:.8f}")


#### if you want to analyze bias/variance in terms of number of MC runs N :

analysisWithN = 0
if analysisWithN:
    Nstart = 100
    Nend = 10000
    Nit = 10
    
    M_coarse = 200  # Number of time steps (1/h)
    #Warning : we are also going to simulate 2M, to compare with finer grid. 
    h_coarse = T / M_coarse  # Step size for discretization
    
    resultScanN = VaryN_MC(S0, r, sigma, K, T, h_coarse, Nstart, Nend, Nit)
    
    # Plot bias vs N:
    plt.figure(figsize=(8, 6))
    plt.loglog(resultScanN['Nvec'], resultScanN['bias'], marker='o', label='Bias')
    constantBias = resultScanN['bias'][0] * np.sqrt(resultScanN['Nvec'][0])
    plt.loglog(resultScanN['Nvec'], constantBias / np.sqrt(resultScanN['Nvec']), label='∝ 1/sqrt(N)', linestyle='--')
    plt.xlabel('Number of MC realisations (N)')
    plt.ylabel('Bias E[Y_0] - E[Y_1]')
    plt.title('Bias scaling with number MC realisations (N), with fixed step size:'+str(h_coarse))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
    
    # Plot variance vs N:
    plt.figure(figsize=(8, 6))
    plt.loglog(resultScanN['Nvec'], resultScanN['var_coarse_estimator'], marker='o', label='Variance')
    constantVar = resultScanN['var_coarse_estimator'][0] * resultScanN['Nvec'][0]
    plt.loglog(resultScanN['Nvec'], constantVar / resultScanN['Nvec'], label='\propto 1/N', linestyle='--')
    plt.xlabel('Number of MC realisations (N)')
    plt.ylabel('Var[Y_0]')
    plt.title('Variance of the estimator scaling with number of MC realisations (N), with fixed step size:'+str(h_coarse))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
 
 #### if you want to analyze bias/variance in terms of number of time step h :
    
analysis_with_h = 1
if analysis_with_h:
    
    #decide which value h_coarse will take :
    hstart = 1/1000
    hend = 1/10
    Nit = 10
    N_MC = 10000 #number of Monte Carlo iterations, this wont change
    
    resultsScan_h = Vary_h_coarse(S0, r, sigma, K, T,  N_MC, hstart, hend, Nit)
    
    # Plot Var(Y1 - Y0) vs h: (Y1 is obtained with the fine grid, and Y0 with
    # the coarse grid)
    plt.figure(figsize=(8, 6))
    plt.plot(resultsScan_h['hvec'], resultsScan_h['var_diff'], marker='o', label='Var(Y1 - Y0)')
    constant = resultsScan_h['var_diff'][-1] / resultsScan_h['hvec'][-1]
    plt.plot(resultsScan_h['hvec'], constant * resultsScan_h['hvec'], label='∝ h', linestyle='--')
    plt.xlabel('Step size (h)')
    plt.ylabel('Var(Y1 - Y0)')
    plt.title('Var(Y1 - Y0) versus step size h, with fixed N_MC = '+str(N_MC))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
    # Plot Esp(Y1 - Y0) vs h: (Y1 is obtained with the fine grid, and Y0 with
    # the coarse grid)
    plt.figure(figsize=(8, 6))
    plt.plot(resultsScan_h['hvec'], resultsScan_h['esp_diff'], marker='o', label='Esp(Y1 - Y0)')
    constant = resultsScan_h['esp_diff'][-1] / resultsScan_h['hvec'][-1]
    plt.plot(resultsScan_h['hvec'], constant * resultsScan_h['hvec'], label='∝ h', linestyle='--')
    plt.xlabel('Step size (h)')
    plt.ylabel('E[Y1 - Y0]')
    plt.title('E[Y1 - Y0] versus step size h, with fixed N_MC = '+str(N_MC))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
    # Plot bias vs h: 
    plt.figure(figsize=(8, 6))
    plt.plot(resultsScan_h['hvec'], resultsScan_h['bias'], marker='o', label='Bias E[Y1] - E[Y0]')
    plt.xlabel('Step size (h)')
    plt.ylabel('Bias')
    plt.title('Bias versus step size h, with fixed N_MC = '+str(N_MC))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
    # Plot Variance of the estimator obtained with the coarse grid, vs h: 
    plt.figure(figsize=(8, 6))
    plt.plot(resultsScan_h['hvec'], resultsScan_h['var_coarse_estimator'], marker='o', label='Var[Y0]')
    plt.xlabel('Step size (h)')
    plt.ylabel('Variance of Y_h0')
    plt.title('Variance versus step size h, with fixed N_MC = '+str(N_MC))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
    
    