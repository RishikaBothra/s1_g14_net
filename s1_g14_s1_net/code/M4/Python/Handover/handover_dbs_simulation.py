"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   handover_dbs_simulation.py  (OPTIMISED VERSION)                   ###
###   Purpose: Monte-Carlo simulation of handover probability for        ###
###            both SSM (constant v) and DSM (Uniform-distributed v).   ###
###            Corresponds to Fig. 2 simulation curves.                 ###
###                                                                     ###
###   Speed optimisations vs original:                                  ###
###       1. Time loop FULLY VECTORISED with numpy broadcasting         ###
###       2. R_UAV shrunk to 3e4 (keeps ~10× more UAVs than needed)    ###
###          while preserving nearest-neighbour statistics               ###
###       3. Realizations = 2 000 (sufficient for smooth curves)        ###
###       4. Params saved to JSON for theory scripts to reuse           ###
###########################################################################
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import sample_params, save_params

# ── 1. Sample + save randomised parameters ───────────────────────────────
params = sample_params(verbose=True)

lambda0UAV = params["lambda0UAV"]
vStart     = params["vStart"]
vEnd       = params["vEnd"]
vMean_ms   = params["vMean_ms"]
T          = params["T"]
dt         = params["dt"]
tVec       = np.array(params["tVec_handover"], dtype=float)  # (T,)

os.makedirs("Data/Mobile_DBS", exist_ok=True)
save_params(params, "Data/handover_run_params.json")

# ── Reduced R_UAV: large enough for valid nearest-neighbour stats ─────────
# Original: 1e5 m → ~37 000 UAVs.  Using 3e4 m keeps ~3 300 UAVs,
# which is still >> 1 (serving DBS always exists) but 10× faster.
R_UAV = 3e4
NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2

tLen = len(tVec)
Realizations = 10_000  # 10K realisations → publication-quality smooth curves

rng = np.random.default_rng()

print(f"\nR_UAV = {R_UAV/1e3:.0f} km  |  "
      f"E[NumUAV] = {NumUAV_Initial:.0f}  |  "
      f"T = {T} s  |  Realisations = {Realizations:,}")

# ── Helper: run one Monte-Carlo block ─────────────────────────────────────
def run_mc(Realizations, speed_fn):
    """
    Fully-vectorised handover MC over tLen time steps.

    speed_fn(NumUAV, rng) → 1-D speed array of length NumUAV.

    Returns HandoverMat (Realizations × tLen, bool).
    """
    HandoverMat = np.zeros((Realizations, tLen), dtype=np.uint8)
    _t0 = time.time()

    for i in range(Realizations):
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - _t0
            eta = elapsed / i * (Realizations - i)
            print(f"      {i}/{Realizations} done  |  ETA: {eta:.0f} s")
        NumUAV = rng.poisson(NumUAV_Initial)
        if NumUAV < 2:   # need at least 2 DBSs for a handover to be possible
            continue

        v = speed_fn(NumUAV, rng)                  # (NumUAV,)

        # Initial PPP positions
        r       = R_UAV * np.sqrt(rng.uniform(0, 1, NumUAV))
        theta   = rng.uniform(0, 2 * np.pi, NumUAV)
        PosUAV  = np.column_stack([r * np.cos(theta), r * np.sin(theta)])   # (N,2)

        # Random movement direction per UAV
        phi     = rng.uniform(0, 2 * np.pi, NumUAV)
        dirVec  = np.column_stack([np.cos(phi), np.sin(phi)])                # (N,2)

        IndMin_init = int(np.argmin(r))

        # ── Vectorised time loop ──────────────────────────────────────────
        # Displacement at each time: vd[t_idx, uav_idx, xy] = v[uav]*t[t]*dir[uav,xy]
        # Shape: (tLen, N, 2)
        vd = (v[None, :, None] * tVec[:, None, None]) * dirVec[None, :, :]

        # New positions: (tLen, N, 2)
        NewPos    = PosUAV[None, :, :] + vd

        # Distance from origin at each time: (tLen, N)
        NewRanges = np.sqrt(np.sum(NewPos**2, axis=2))

        # Serving DBS index at each time: (tLen,)
        IndMins   = np.argmin(NewRanges, axis=1)

        # Handover = serving DBS changed
        HandoverMat[i, :] = (IndMins != IndMin_init).astype(np.uint8)

    return HandoverMat


# ── 2. DSM simulation (Uniform random speed) ─────────────────────────────
print("\n[1/2] DSM simulation (Uniform speed)…")
t0 = time.time()
mat_DSM = run_mc(Realizations,
                 lambda N, rng: rng.uniform(vStart, vEnd, N))
print(f"      Done in {time.time()-t0:.1f} s")

# ── 3. SSM simulation (constant speed = vMean_ms) ─────────────────────────
print("[2/2] SSM simulation (Constant speed)…")
t0 = time.time()
mat_SSM = run_mc(Realizations,
                 lambda N, rng: np.full(N, vMean_ms))
print(f"      Done in {time.time()-t0:.1f} s")

# ── 4. Compute & save curves ───────────────────────────────────────────────
Handover_Uniform_DSM_Simulation  = mat_DSM.mean(axis=0)
Handover_Constant_SSM_Simulation = mat_SSM.mean(axis=0)

np.save("Data/Mobile_DBS/Handover_UniformVelocityDBS_Simulation.npy",
        Handover_Uniform_DSM_Simulation)
np.save("Data/Mobile_DBS/Handover_ConstantVelocityDBS_Simulation.npy",
        Handover_Constant_SSM_Simulation)

print("\nSaved → Data/Mobile_DBS/Handover_UniformVelocityDBS_Simulation.npy")
print("Saved → Data/Mobile_DBS/Handover_ConstantVelocityDBS_Simulation.npy")
print(f"\nHandover_Uniform shape : {Handover_Uniform_DSM_Simulation.shape}")
print(f"Handover_Constant shape: {Handover_Constant_SSM_Simulation.shape}")
