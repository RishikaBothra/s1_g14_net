"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   plot_handover.py  (IMPROVED)                                      ###
###   Purpose: Fig. 2 – handover probability SSM vs DSM.               ###
###            All available curves plotted as smooth lines.            ###
###            Uses scipy smoothing for noisy MC data.                  ###
###                                                                     ###
###   Run AFTER handover_dbs_simulation.py and                          ###
###             handover_theory_uniform_lowerbound.py                   ###
###########################################################################
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d
import os

# ── Use a clean, paper-quality style ─────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   12,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

DATA_DIR = "Data/Mobile_DBS"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("Data", exist_ok=True)

# ── Load all available curves (missing files are silently skipped) ────────
def load(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return None
    return np.load(path)

SSM_Theory    = load("Handover_ConstantVelocityDBS_Theory.npy")
SSM_Sim       = load("Handover_ConstantVelocityDBS_Simulation.npy")
DSM_Ray_Theory= load("Handover_RayleighVelocityDBS_Theory.npy")
DSM_Unif_Sim  = load("Handover_UniformVelocityDBS_Simulation.npy")
DSM_Unif_Th   = load("Handover_Uniform40VelocityDBS_Theory.npy")

# ── Build time axis ───────────────────────────────────────────────────────
available = [x for x in [SSM_Theory, SSM_Sim, DSM_Ray_Theory,
                          DSM_Unif_Sim, DSM_Unif_Th] if x is not None]
if not available:
    print("ERROR: No data files found. Run simulation scripts first.")
    exit()

T    = max(len(x) for x in available)
tVec = np.arange(1, T + 1)

# ── Smooth helper ────────────────────────────────────────────────────────
def smooth(arr, sigma=20):
    """
    Two-step smoothing for handover probability curves:
      1. Cumulative maximum  → enforces strict monotonicity
         (P_H can never decrease once a handover has occurred)
      2. Gaussian filter (sigma=12) → removes MC noise
    """
    y = np.asarray(arr, dtype=float)
    y = np.maximum.accumulate(y)          # monotone non-decreasing
    y = gaussian_filter1d(y, sigma=sigma) # smooth
    return np.clip(y, 0.0, 1.0)

# ── Colour / style scheme ─────────────────────────────────────────────────
COLORS = {
    "dsm_sim"   : "#2ca02c",   # green
    "dsm_unif_th": "#17becf",  # teal
    "dsm_ray_th": "#9467bd",   # purple
    "ssm_sim"   : "#d62728",   # red
    "ssm_th"    : "#1f77b4",   # blue
}

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim([0, T])
ax.set_ylim([0, 1.05])
ax.grid(True, linestyle="--", alpha=0.35)

# Helper to safely plot a full curve
def plot_curve(data, color, ls, label, smooth_sigma=3):
    if data is None:
        return
    t = tVec[:len(data)]
    y = smooth(data, smooth_sigma) if smooth_sigma > 0 else data
    y = np.clip(y, 0, 1)
    ax.plot(t, y, color=color, linestyle=ls, linewidth=2.0, label=label)

# DSM curves (variable speed)
plot_curve(DSM_Unif_Sim,   COLORS["dsm_sim"],    "-",  "DSM, Simulation (Uniform $v$)")
plot_curve(DSM_Unif_Th,    COLORS["dsm_unif_th"],"--", "DSM, Theory – Lower Bound (Uniform $v$)")
plot_curve(DSM_Ray_Theory, COLORS["dsm_ray_th"], ":",  "DSM, Theory – Lower Bound (Rayleigh $v$)")

# SSM curves (constant speed)  — plotted as SMOOTH LINE, not just markers
plot_curve(SSM_Sim,  COLORS["ssm_sim"], "-.",  "SSM, Simulation (Constant $v$)")
plot_curve(SSM_Theory, COLORS["ssm_th"], "--", "SSM, Theory – Exact (Constant $v$)")

# ── Add a few markers on SSM simulation so it stays paper-style ──────────
if SSM_Sim is not None:
    lim = min(len(SSM_Sim), T)
    marker_idx = np.linspace(0, lim - 1, 12, dtype=int)
    ax.plot(tVec[marker_idx],
            np.clip(smooth(SSM_Sim)[marker_idx], 0, 1),
            color=COLORS["ssm_sim"], linestyle="none",
            marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.8)

ax.set_xlabel("Time $t$ (s)", fontsize=14)
ax.set_ylabel(r"$\mathrm{P}[H(t)]$", fontsize=14)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")
ax.set_title("Handover Probability: SSM vs DSM  [Randomised Parameters]",
             fontsize=12, pad=10)

plt.tight_layout()
plt.savefig("Data/Fig2_Handover.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → Data/Fig2_Handover.png")
