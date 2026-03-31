"""
Paper: Handover Probability in Drone Cellular Networks
Authors: Morteza Banagar, Vishnu V. Chetlur, and Harpreet S. Dhillon
Emails: mbanagar@vt.edu, vishnucr@vt.edu, hdhillon@vt.edu

This code is used to generate the simulation data for Fig. 1, 
density of the network of non-serving DBSs for the DSM with 
Rayleigh distributed speed.
"""

import numpy as np
import scipy.io as sio
from datetime import datetime
import multiprocessing as mp
import time

# Simulation Parameters

lambda0UAV = 1e-6 # drone density 

R_UAV = 1e4 # maximum range of the drone network

NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2 # Expected number of DBSs inside the simulation area.

vMean = 45 # [km/h] # Average speed of the drone, which is used to calculate the scale parameter for the Rayleigh distribution.

vMean = vMean / 3.6 # [m/s] 

vSigma = vMean * np.sqrt(2 / np.pi) # Scale parameter for the Rayleigh distribution, derived from the mean speed.

dr = 1 # Step size for the range bins in the simulation, which determines how the distances are discretized for counting the number of DBSs.    

NumR = int(np.round((R_UAV - dr) / dr) + 1) # Number of range bins, calculated based on the maximum range and step size. This determines how many discrete distance intervals are considered in the simulation.

u0 = 800 # Initial distance of the UE and the location of the serving DBS.

tVec = np.array([10, 20, 40, 100]) # Condition: v * t < R_UAV

tLen = len(tVec) # # Number of time values being evaluated. This is used to determine the dimensions of the output density matrix and to loop through the different time values during the simulation.

Realizations = int(1e5) # Number of realizations for the Monte Carlo simulation

def simulate_chunk(realizations_chunk):
    """Function to process a chunk of realizations for parallel processing."""
    CountPointsAll = np.zeros((tLen, NumR))
    
    for _ in range(realizations_chunk):
        CountPoints = np.zeros((tLen, NumR))
        NumUAV = np.random.poisson(NumUAV_Initial)
        
        if NumUAV == 0:
            continue
            
        PosUAV_Range = np.random.uniform(0, 1, NumUAV)
        PosUAV_Range = R_UAV * np.sqrt(PosUAV_Range)
        PosUAV_Theta = np.random.uniform(0, 2 * np.pi, NumUAV)
        
        # Filter points inside u0
        valid_idx = PosUAV_Range > u0
        PosUAV_Range = PosUAV_Range[valid_idx]
        PosUAV_Theta = PosUAV_Theta[valid_idx]
        NumUAV = len(PosUAV_Range)
        
        if NumUAV == 0:
            continue
            
        v = np.random.rayleigh(scale=vSigma, size=NumUAV)
        PosUAV_X = PosUAV_Range * np.cos(PosUAV_Theta)
        PosUAV_Y = PosUAV_Range * np.sin(PosUAV_Theta)
        PosUAV = np.column_stack((PosUAV_X, PosUAV_Y))
        
        DisplacedTheta = np.random.uniform(0, 2 * np.pi, NumUAV)
        
        for kk, t in enumerate(tVec):
            vd_X = v * t * np.cos(DisplacedTheta)
            vd_Y = v * t * np.sin(DisplacedTheta)
            vd = np.column_stack((vd_X, vd_Y))
            
            DisplacedPosUAV = PosUAV + vd
            NewRange = np.sqrt(np.sum(DisplacedPosUAV**2, axis=1))
            SlottedRange = np.ceil(NewRange / dr)
            
            # Count occurrences 
            unique_vals, counts = np.unique(SlottedRange, return_counts=True)
            
            # Keep indices within the valid bounds
            valid_bins = (unique_vals >= 1) & (unique_vals <= NumR)
            unique_vals = unique_vals[valid_bins].astype(int) - 1 # 0-index adjustment
            counts = counts[valid_bins]
            
            CountPoints[kk, unique_vals] = counts
            
        CountPointsAll += CountPoints
        
    return CountPointsAll

if __name__ == '__main__':
    print(f"Simulation started at: {datetime.now()}")
    start_time = time.time()
    
    # Setup Multiprocessing
    cores = mp.cpu_count()
    chunk_size = Realizations // cores
    chunks = [chunk_size] * cores
    chunks[-1] += Realizations % cores  # Distribute remainder to the last core
    
    # Running the Monte Carlo simulation in parallel using multiple CPU cores to make the simulation much faster
    with mp.Pool(cores) as pool:
        results = pool.map(simulate_chunk, chunks)
        
    CountPointsAll = sum(results)
    
    print(f"Loop completed in {time.time() - start_time:.2f} seconds.")
    
    # Calculate Density
    r_vec = np.arange(0, R_UAV, dr)
    AreaAnnulus = np.pi * (2 * r_vec * dr + dr**2)
    CountPointsAll = CountPointsAll / Realizations
    
    # Tile AreaAnnulus to match matrix dimensions
    AreaAnnulus_Matrix = np.tile(AreaAnnulus, (tLen, 1))
    Density_Simulation = CountPointsAll / AreaAnnulus_Matrix
    
    # Save the data
    sio.savemat('Density_RayleighVelocity_Simulation.mat', {'Density_Simulation': Density_Simulation})
    print(f"Simulation completed at: {datetime.now()}")