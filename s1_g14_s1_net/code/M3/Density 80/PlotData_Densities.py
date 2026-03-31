"""
Paper: Handover Probability in Drone Cellular Networks
Authors: Morteza Banagar, Vishnu V. Chetlur, and Harpreet S. Dhillon
Emails: mbanagar@vt.edu, vishnucr@vt.edu, hdhillon@vt.edu

This code is used to generate the plot of Fig. 1, density of 
the network of non-serving DBSs for the DSM with Rayleigh 
distributed speed.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

ClipData1 = 3000

# Load data (ensure the .mat files are in the same directory or update the paths)
try:
    sim_data = sio.loadmat('Density_RayleighVelocity_Simulation.mat')
    Density_Simulation = sim_data['Density_Simulation'][:, :ClipData1]
except FileNotFoundError:
    print("Simulation data not found. Please run the simulation script first.")
    exit()

try:
    theory_data = sio.loadmat('Density_RayleighVelocity_Theory.mat')
    Density_Theory = theory_data['Density_Theory'][:, :ClipData1]
except FileNotFoundError:
    print("Theory data not found. Please run the theory script first.")
    exit()

rVec1 = np.arange(1, ClipData1 + 1)

# MATLAB indices: [100:200:500, 800, 1000, 1300:300:ClipData1]
# Convert to Python lists and subtract 1 to adjust for 0-based indexing
rVecDS_list = list(range(100, 501, 200)) + [800, 1000] + list(range(1300, ClipData1 + 1, 300))
rVecDS = np.array(rVecDS_list) - 1 

MarkerSize = 5
LineWidth = 2

fig, ax = plt.subplots(figsize=(8, 6), num=501)
ax.grid(True)
ax.set_box_aspect(None)

# Plot Theory (Lines) and Simulation (Markers)
# For the legend, we only label the first pair to avoid duplicate legend entries
ax.plot(rVec1, Density_Theory[0, :], 'b-', linewidth=LineWidth, label='Theory')
ax.plot(rVecDS + 1, Density_Simulation[0, rVecDS], 'ro', markersize=MarkerSize, 
        markerfacecolor='w', linewidth=LineWidth, label='Simulation')

# Plot remaining t entries without labels
for i in range(1, 4):
    ax.plot(rVec1, Density_Theory[i, :], 'b-', linewidth=LineWidth)
    ax.plot(rVecDS + 1, Density_Simulation[i, rVecDS], 'ro', markersize=MarkerSize, 
            markerfacecolor='w', linewidth=LineWidth)

# Formatting
ax.set_xlabel(r'$u_{\mathbf{x}}$ (m)', fontsize=14, fontname='Times New Roman')
ax.set_ylabel(r'$\lambda(t; u_{\mathbf{x}}, u_0)$', fontsize=14, fontname='Times New Roman')
ax.tick_params(axis='both', which='major', labelsize=14)

ax.legend(loc='lower right', fontsize=14, frameon=True)

# Annotation properties tailored for Matplotlib
ax.annotate(r'Increasing $t \in \{10, 20, 40, 100\}$ s',
            xy=(1100, 0.25e-6), xycoords='data',
            xytext=(1100, 0.8e-6), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            fontsize=15, ha='center', va='top')

ax.set_ylim([0, 1.05e-6])

plt.tight_layout()
plt.show()