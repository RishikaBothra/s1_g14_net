import numpy as np
import scipy.io as sio
import os
from datetime import datetime

# --- Parameters ---
vDiff_kmh = 40  # Change this to 80 for the other Uniform case
lambda0UAV = 1e-6
vMean = 45 / 3.6
vDiff = vDiff_kmh / 3.6 
vStart, vEnd = vMean - vDiff/2, vMean + vDiff/2
dt, T = 1, 300
tVec = np.arange(dt, T + dt, dt)
Realizations = int(1e5)

print(f"Starting Monte Carlo Theory calculation for Uniform {vDiff_kmh}...")

# Vectorized simulation approach to approximate theory
HandoverMat = np.zeros((Realizations, len(tVec)))

for i in range(Realizations):
    # Sample single nearest DBS characteristics
    # u0 is sampled based on the PDF of the nearest neighbor distance in a PPP
    u0 = np.sqrt(np.random.uniform(0, 1)) * (1 / np.sqrt(np.pi * lambda0UAV))
    v = np.random.uniform(vStart, vEnd)
    theta0 = np.random.uniform(0, 2 * np.pi)
    
    # Calculate handover probability for this sample
    # Formula: P(ho) = 1 - exp(-pi * lambda * distance^2)
    dist_sq = u0**2 + (v*tVec)**2 - 2*u0*v*tVec*np.cos(theta0)
    HandoverMat[i, :] = 1 - np.exp(-np.pi * lambda0UAV * dist_sq)

Handover_UniformTheory = np.mean(HandoverMat, axis=0)

# --- Saving Logic ---
root_dir = os.path.dirname(os.getcwd())
output_folder = os.path.join(root_dir, 'Data', 'Mobile DBS')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_name = f'Handover_Uniform{vDiff_kmh}VelocityDBS_Theory.mat'
var_name = f'Handover_Uniform{vDiff_kmh}DBS_Theory'
file_path = os.path.join(output_folder, file_name)

sio.savemat(file_path, {var_name: Handover_UniformTheory})

print(f"Theory data saved to: {file_path}")
print(f"Finished at: {datetime.now()}")