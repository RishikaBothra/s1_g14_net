import numpy as np
import scipy.io as sio
import os
from scipy.integrate import tplquad
from scipy.stats import rayleigh
from datetime import datetime

# --- Parameters ---
lambda0UAV = 1e-6
vMean = 45 / 3.6
vSigma = vMean * np.sqrt(2 / np.pi)
u0Sigma = 1 / np.sqrt(2 * np.pi * lambda0UAV)

fv = lambda v: rayleigh.pdf(v, scale=vSigma)
fu0 = lambda u: rayleigh.pdf(u, scale=u0Sigma)

def integrand(theta0, u0, v, t):
    # Kernel for variable velocity lower bound
    dist_sq = u0**2 + (v*t)**2 - 2*u0*v*t*np.cos(theta0)
    return np.exp(-2 * np.pi * lambda0UAV * dist_sq) * fv(v) * fu0(u0) * (1/(2*np.pi))

# Full time vector for the plot (1 to 300 seconds)
tVec = np.arange(1, 301, 1) 
Handover_RayleighDBS_Theory = []

print(f"Theory calculation started: {datetime.now()}")

for t in tVec:
    # Triple integral over v, u0, and theta
    # Limits: v [0, 3*vSigma], u0 [0, 3*u0Sigma], theta [0, 2*pi]
    q, _ = tplquad(integrand, 0, 3*vSigma, 
                   lambda v: 0, lambda v: 3*u0Sigma,
                   lambda v, u0: 0, lambda v, u0: 2*np.pi, 
                   args=(t,))
    Handover_RayleighDBS_Theory.append(1 - q)

# --- Saving Logic ---
# Getting root folder via os.getcwd (assuming you are in 'Mobile DBS' folder)
root_dir = os.path.dirname(os.getcwd())
output_folder = os.path.join(root_dir, 'Data', 'Mobile DBS')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_path = os.path.join(output_folder, 'Handover_RayleighVelocityDBS_Theory.mat')
sio.savemat(file_path, {'Handover_RayleighDBS_Theory': Handover_RayleighDBS_Theory})

print(f"Theory data saved to: {file_path}")
print(f"Finished at: {datetime.now()}")