"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   plot_densities.py                                                 ###
###   Purpose: Reproduce Fig. 1 – spatial density of non-serving DBSs. ###
###            Loads pre-computed .npy files and plots Theory vs Sim.   ###
###                                                                     ###
###   Run AFTER:                                                         ###
###       python density_rayleigh_theory.py                             ###
###       python density_rayleigh_simulation.py                         ###
###########################################################################
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.makedirs("Data", exist_ok=True)

# ── Load pre-computed data ────────────────────────────────────────────────
Density_Simulation = np.load("Data/Density_RayleighVelocity_Simulation.npy")
Density_Theory     = np.load("Data/Density_RayleighVelocity_Theory.npy")

ClipData = 3000   # clip to first 3 000 m  (same as MATLAB PlotData_Densities)
Density_Simulation = Density_Simulation[:, :ClipData]
Density_Theory     = Density_Theory[:, :ClipData]

r_vec = np.arange(1, ClipData + 1)   # 1 … 3000 m

# Marker positions matching the MATLAB script
rVecDS = np.concatenate([
    np.arange(100, 601, 200),
    [800, 1000],
    np.arange(1300, ClipData + 1, 300),
])
rVecDS = rVecDS[rVecDS <= ClipData].astype(int) - 1   # 0-indexed

tVec = [10, 20, 40, 100]
tLen = len(tVec)

# ── Dynamic y-limit ───────────────────────────────────────────────────────
# FIX: Old hardcoded 1.05e-6 clips curves when randomised lambda0UAV > 1e-6.
# Auto-scale to the actual max value in the theory data.
finite_theory = Density_Theory[np.isfinite(Density_Theory)]
y_max = float(np.max(finite_theory)) * 1.10 if len(finite_theory) > 0 else 2e-6

# ── Colour palettes ───────────────────────────────────────────────────────
theory_colors = plt.cm.Blues(np.linspace(0.40, 0.95, tLen))
# Distinct colours per time step for simulation markers
sim_colors = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572"]

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

for kk in range(tLen):
    ax.plot(r_vec, Density_Theory[kk, :],
            color=theory_colors[kk], linewidth=2,
            label=f"Theory $t={tVec[kk]}$ s")

    # Simulation markers colour-coded per t so each time step is visible
    ax.plot(r_vec[rVecDS], Density_Simulation[kk, rVecDS],
            color=sim_colors[kk], linewidth=0,
            marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.8,
            label=f"Sim $t={tVec[kk]}$ s")

ax.set_xlabel(r"$u_\mathbf{x}$ (m)", fontsize=14)
ax.set_ylabel(r"$\lambda(t;\,u_\mathbf{x},u_0)$", fontsize=14)
ax.set_ylim([0, y_max])
ax.set_xlim([0, ClipData])
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)

# Annotation scaled to dynamic y_max
ax.annotate(
    r"Increasing $t \in \{10,20,40,100\}$ s",
    xy=(180, y_max * 0.85),
    xytext=(700, y_max * 0.30),
    fontsize=11,
    arrowprops=dict(arrowstyle="->", lw=1.5),
)

plt.title("Spatial Density of Non-Serving DBSs (DSM, Rayleigh Speed)\n"
          "[Randomised Parameters]", fontsize=12)
plt.tight_layout()
plt.savefig("Data/Fig1_Densities.png", dpi=150)
plt.show()
print("Saved → Data/Fig1_Densities.png")
