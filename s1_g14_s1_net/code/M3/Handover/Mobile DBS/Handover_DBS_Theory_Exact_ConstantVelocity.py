import numpy as np
import scipy.io as sio
import os
from scipy.integrate import dblquad
from scipy.stats import rayleigh
from datetime import datetime

# --- Parameters ---
lambda0UAV = 1e-6
v = 45 / 3.6
dt, T = 1, 300
tVec = np.arange(dt, T + dt, dt)
u0Sigma = 1 / np.sqrt(2 * np.pi * lambda0UAV)
fu0 = lambda u: rayleigh.pdf(u, scale=u0Sigma)

def Fun1(u0, theta0, v, t):
    # Geometric kernel for the handover integration
    val = (u0**2 + (v*t)**2 - 2*u0*v*t*np.cos(theta0))
    return val

def compute_theory(t):
    # Probability that initial serving BS is still the nearest
    # We integrate over u0 and theta0
    fun0 = lambda theta0, u0: np.exp(-2 * np.pi * lambda0UAV * Fun1(u0, theta0, v, t)) * \
                             fu0(u0) * (1 / (2 * np.pi))
    
    # Integration limits: u0 from 0 to 5*u0Sigma (approximating infinity), theta from 0 to 2*pi
    q, _ = dblquad(fun0, 0, 5*u0Sigma, lambda x: 0, lambda x: 2*np.pi)
    return 1 - q

print(f"Computing Constant Velocity Theory (this may take a moment)...")
start_time = datetime.now()

# Compute theory for all t
results = [compute_theory(t) for t in tVec]

# --- Saving Logic ---
# Get root directory and navigate to Data/Mobile DBS
root_dir = os.path.dirname(os.getcwd())
output_folder = os.path.join(root_dir, 'Data', 'Mobile DBS')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_path = os.path.join(output_folder, 'Handover_ConstantVelocityDBS_Theory.mat')
sio.savemat(file_path, {'Handover_ConstantDBS_Theory': results})

print(f"Theory calculation finished in {datetime.now() - start_time}")
print(f"Saved to: {file_path}")