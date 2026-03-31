"""
Paper: Handover Probability in Drone Cellular Networks
Authors: Morteza Banagar, Vishnu V. Chetlur, and Harpreet S. Dhillon
Emails: mbanagar@vt.edu, vishnucr@vt.edu, hdhillon@vt.edu

This code is used to generate the theoretical data for Fig. 1, 
density of the network of non-serving DBSs for the DSM with 
Rayleigh distributed speed.
"""

import numpy as np
import scipy.io as sio
import scipy.integrate as integrate
import scipy.stats as stats
from datetime import datetime
import time

print(f"Theory calculation started at: {datetime.now()}")

lambda0UAV = 1e-6
R_UAV = 1e4
NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2
vMean = 80 # [km/h]
vMean = vMean / 3.6 # [m/s]
vSigma = vMean * np.sqrt(2 / np.pi)
dr = 1
NumR = int(np.round((R_UAV - dr) / dr) + 1)
u0 = 800
tVec = [10, 20, 40, 100]
tLen = len(tVec)

fv = lambda v1: stats.rayleigh.pdf(v1, scale=vSigma)
Fv = lambda v1: stats.rayleigh.cdf(v1, scale=vSigma)

Density_Theory = np.zeros((tLen, NumR))

for kk, t in enumerate(tVec):
    start_time = time.time()
    
    # arange behaves like start:step:stop. Add dr to stop to make it inclusive like MATLAB
    ux_values = np.arange(dr, round(R_UAV) + dr, dr) 
    
    for ux in ux_values:
        if ux > R_UAV: # boundary check
            break
            
        def fun1(vi):
            # Calculate argument for arccos and clip to [-1, 1] to avoid math domain errors
            arg = (vi**2 * t**2 + ux**2 - u0**2) / (2 * vi * t * ux)
            arg = np.clip(arg, -1.0, 1.0)
            return fv(vi) * 1/np.pi * np.arccos(arg)

        A = 1 - Fv(max(0, (u0 - ux) / t))
        
        lower_limit = abs(u0 - ux) / t
        upper_limit = (u0 + ux) / t
        
        B, _ = integrate.quad(fun1, lower_limit, upper_limit, limit=100)
        
        idx = int(round(ux / dr)) - 1 # 0-indexed adjustment
        if 0 <= idx < NumR:
            Density_Theory[kk, idx] = lambda0UAV * (A - B)
            
    print(f"t={t} completed in {time.time() - start_time:.2f} seconds.")

sio.savemat('Density_RayleighVelocity_Theory.mat', {'Density_Theory': Density_Theory})
print(f"Theory calculation completed at: {datetime.now()}")