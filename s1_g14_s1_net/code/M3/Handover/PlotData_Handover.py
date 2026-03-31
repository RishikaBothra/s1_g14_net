import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# Define the data directory path based on your provided structure
data_dir = os.path.join('.', 'Data', 'Mobile DBS')

def load_mat_file(file_name):
    path = os.path.join(data_dir, file_name)
    try:
        return sio.loadmat(path)
    except FileNotFoundError:
        print(f"Warning: {file_name} not found in {data_dir}")
        return None

# Load all simulation and theory datasets
data_const_sim = load_mat_file('Handover_ConstantVelocityDBS_Simulation.mat')
data_const_theo = load_mat_file('Handover_ConstantVelocityDBS_Theory.mat')
data_rayl_sim = load_mat_file('Handover_RayleighVelocityDBS_Simulation.mat')
data_rayl_theo = load_mat_file('Handover_RayleighVelocityDBS_Theory.mat')
data_uni40_sim = load_mat_file('Handover_Uniform40VelocityDBS_Simulation.mat')
data_uni40_theo = load_mat_file('Handover_Uniform40VelocityDBS_Theory.mat')

# Define range vectors
rVec = np.arange(1, 301) # 1 to 300
# MATLAB: [1 : 10 : 71, 91 : 30 : 300]
rVecDS = np.concatenate([np.arange(1, 72, 10), np.arange(91, 301, 30)])
# Adjust for 0-based indexing in Python for selecting simulation points
rVecDS_idx = rVecDS - 1

# Plot Configuration
MarkerSize = 5
LineWidth = 2

plt.figure(figsize=(10, 7), num=502)
ax = plt.gca()
plt.grid(True)

# 1. Rayleigh Velocity Plot
if data_rayl_theo and data_rayl_sim:
    plt.plot(rVec, data_rayl_theo['Handover_RayleighDBS_Theory'].flatten(), 
             'm:', linewidth=LineWidth, label='Rayleigh Theory')
    plt.plot(rVec, data_rayl_sim['Handover_RayleighDBS_Simulation'].flatten(), 
             'g-', linewidth=LineWidth, label='Rayleigh Simulation')

# 2. Constant Velocity Plot
if data_const_theo and data_const_sim:
    plt.plot(rVec, data_const_theo['Handover_ConstantDBS_Theory'].flatten(), 
             'b--', linewidth=LineWidth, label='Constant Theory')
    sim_vals = data_const_sim['Handover_ConstantDBS_Simulation'].flatten()
    plt.plot(rVecDS, sim_vals[rVecDS_idx], 
             'ro', markersize=MarkerSize, markerfacecolor='w', 
             linestyle='None', markeredgewidth=LineWidth, label='Constant Simulation')


# 3. Uniform (40) Velocity Plot
if data_uni40_theo and data_uni40_sim:
    # Plot Theory as a line
    plt.plot(rVec, data_uni40_theo['Handover_Uniform40DBS_Theory'].flatten(), 
             'k--', linewidth=LineWidth, label='Uniform 40 Theory') # Changed to black dashed
    
    # Plot Simulation as markers only
    uni40_sim_vals = data_uni40_sim['Handover_Uniform40DBS_Simulation'].flatten()
    plt.plot(rVecDS, uni40_sim_vals[rVecDS_idx], 
             'bs', markersize=MarkerSize, markerfacecolor='w', 
             linestyle='None', markeredgewidth=LineWidth, label='Uniform 40 Simulation') # 'bs' = blue squares
    
# Labels and Formatting
plt.xlabel(r'$u_\mathbf{x}$ (m)', fontsize=14)
plt.ylabel('Handover Probability', fontsize=14)
plt.ylim([0, 1]) # Handover probability is between 0 and 1
plt.legend(loc='lower right', fontsize=12)

# Set font to match the "Times" requirement from MATLAB
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.tight_layout()
plt.show()