"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   density_rayleigh_simulation.py                                    ###
###   Purpose: Monte-Carlo simulation of the spatial density of         ###
###            non-serving DBSs (DSM, Rayleigh-distributed speed).      ###
###            Corresponds to Fig. 1 simulation data.                   ###
###                                                                     ###
###   Randomised algorithm: all key physical parameters (UAV density,   ###
###   mean speed, exclusion radius) are drawn from distributions via    ###
###   randomised_config.sample_params() before the simulation begins.   ###
###########################################################################
"""

import sys, os, time
import numpy as np
# NOTE: Rayleigh sampling done via numpy inverse-CDF (avoids scipy Generator
# compatibility issues across scipy versions).  If U~Uniform(0,1) then
# X = sigma * sqrt(-2*ln(U))  ~  Rayleigh(sigma)

# Allow importing randomised_config from the parent Python/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import load_params

# ── 1. Load the SAME parameters as the theory script ────────────────────
# density_rayleigh_theory.py saves params to Data/density_run_params.json.
# Loading here ensures theory and simulation use IDENTICAL randomised values.
params = load_params("Data/density_run_params.json", verbose=True)

lambda0UAV = params["lambda0UAV"]
R_UAV      = params["R_UAV_density"]
vSigma     = params["vSigma"]
u0         = params["u0"]
tVec       = params["tVec_density"]
dr         = params["dr"]

# ── 2. Derived geometry ───────────────────────────────────────────────────
NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2   # expected number of UAVs
NumR           = int(round((R_UAV - dr) / dr)) + 1
tLen           = len(tVec)
Realizations   = int(1e5)   # reduced from 1e6 for practical Python runtime

os.makedirs("Data", exist_ok=True)  # ensure output directory exists

print(f"\nRunning {Realizations:,} Monte-Carlo realisations …")
t_start = time.time()

rng = np.random.default_rng()   # fresh Generator (non-deterministic seed)

CountPointsAll = np.zeros((tLen, NumR), dtype=np.float64)

for i in range(Realizations):
    # ── Draw number of UAVs from Poisson distribution ────────────────────
    NumUAV = rng.poisson(NumUAV_Initial)

    # ── Draw initial UAV positions (PPP inside disk of radius R_UAV) ─────
    # Radial position: use inverse CDF of f_r(r)=2r/R² → r = R*sqrt(U[0,1])
    PosUAV_Range = R_UAV * np.sqrt(rng.uniform(0, 1, NumUAV))
    PosUAV_Theta = rng.uniform(0, 2 * np.pi, NumUAV)

    # Remove UAVs inside exclusion zone (range ≤ u0)
    outside = PosUAV_Range > u0
    PosUAV_Range = PosUAV_Range[outside]
    PosUAV_Theta = PosUAV_Theta[outside]
    NumUAV       = len(PosUAV_Range)

    if NumUAV == 0:
        continue

    # ── Draw Rayleigh-distributed speeds (inverse-CDF method) ───────────
    # vSigma is itself randomised via sample_params()
    # Using inverse-CDF: X = vSigma * sqrt(-2 * ln(U)),  U ~ Uniform(0,1)
    # This is mathematically equivalent to Rayleigh(vSigma) and works with
    # all numpy/scipy versions without random_state compatibility concerns.
    u_rayleigh = rng.uniform(0, 1, NumUAV)
    v = vSigma * np.sqrt(-2.0 * np.log(np.clip(u_rayleigh, 1e-15, 1.0)))

    # Cartesian initial positions
    PosUAV = np.column_stack([
        PosUAV_Range * np.cos(PosUAV_Theta),
        PosUAV_Range * np.sin(PosUAV_Theta),
    ])

    # ── Randomised displacement direction (uniform on [0, 2π]) ───────────
    DisplacedTheta = rng.uniform(0, 2 * np.pi, NumUAV)

    for kk, t in enumerate(tVec):
        # Displacement vector
        vd = (v * t)[:, None] * np.column_stack([
            np.cos(DisplacedTheta), np.sin(DisplacedTheta)
        ])
        DisplacedPosUAV = PosUAV + vd
        NewRange        = np.sqrt(np.sum(DisplacedPosUAV**2, axis=1))

        # Bin into annular rings of width dr using fast bincount (vectorised)
        SlottedRange = np.ceil(NewRange / dr).astype(int) - 1  # 0-indexed

        # FIX: DISCARD UAVs that moved outside R_UAV — don't clip them into
        # the last bin (clip was under-counting interior bins by accumulating
        # all boundary-exceeding particles in the outermost cell).
        valid = (SlottedRange >= 0) & (SlottedRange < NumR)
        counts = np.bincount(SlottedRange[valid], minlength=NumR)
        CountPointsAll[kk] += counts

elapsed = time.time() - t_start
print(f"Simulation completed in {elapsed:.1f} s")

# ── 3. Normalise to obtain density ───────────────────────────────────────
AreaAnnulus = np.pi * (2 * np.arange(0, R_UAV, dr) * dr + dr**2)  # shape (NumR,)

CountPointsAll   /= Realizations
Density_Simulation = CountPointsAll / AreaAnnulus[np.newaxis, :]   # (tLen, NumR)

# ── 4. Save results ───────────────────────────────────────────────────────
np.save("Data/Density_RayleighVelocity_Simulation.npy", Density_Simulation)
print("Saved → Data/Density_RayleighVelocity_Simulation.npy")
print(f"\nDensity shape: {Density_Simulation.shape}  (tLen={tLen}, NumR={NumR})")
