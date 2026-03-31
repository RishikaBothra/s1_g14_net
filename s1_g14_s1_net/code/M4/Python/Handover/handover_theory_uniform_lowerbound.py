"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   handover_theory_uniform_lowerbound.py  (OPTIMISED)                ###
###   Purpose: Simulate lower-bound handover probability for DSM        ###
###            with Uniform-distributed DBS speed (Monte-Carlo).        ###
###            Corresponds to DSM Uniform curves in Fig. 2.             ###
###                                                                     ###
###   Speed optimisations:                                              ###
###       1. Fully vectorised time loop (numpy broadcasting)            ###
###       2. Same R_UAV = 3e4 m as simulation script                    ###
###       3. Realizations = 2 000                                       ###
###########################################################################
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import load_params

# ── 1. Load the SAME parameters as the simulation script ──────────────────
params     = load_params("Data/handover_run_params.json", verbose=True)
lambda0UAV = params["lambda0UAV"]
vStart     = params["vStart"]
vEnd       = params["vEnd"]
T          = params["T"]
dt         = params["dt"]
tVec       = np.array(params["tVec_handover"], dtype=float)   # (T,)
tLen       = len(tVec)

R_UAV = 3e4   # match handover_dbs_simulation.py
NumUAV_Initial = lambda0UAV * np.pi * R_UAV**2
Realizations   = 10_000  # 10K → publication-quality

rng = np.random.default_rng()

print(f"\nDSM Uniform lower bound | R_UAV = {R_UAV/1e3:.0f} km | "
      f"E[NumUAV] = {NumUAV_Initial:.0f} | T = {T} s")
t_start = time.time()

HandoverMat = np.zeros((Realizations, tLen), dtype=np.uint8)

for i in range(Realizations):
    if i > 0 and i % 500 == 0:
        elapsed_so_far = time.time() - t_start
        eta = elapsed_so_far / i * (Realizations - i)
        print(f"  {i}/{Realizations} done  |  ETA: {eta:.0f} s")
    NumUAV = rng.poisson(NumUAV_Initial)
    if NumUAV < 2:
        continue

    # Uniform random speeds
    v = rng.uniform(vStart, vEnd, NumUAV)

    # PPP positions
    r      = R_UAV * np.sqrt(rng.uniform(0, 1, NumUAV))
    theta  = rng.uniform(0, 2 * np.pi, NumUAV)
    PosUAV = np.column_stack([r * np.cos(theta), r * np.sin(theta)])   # (N,2)

    phi    = rng.uniform(0, 2 * np.pi, NumUAV)
    dirVec = np.column_stack([np.cos(phi), np.sin(phi)])                # (N,2)

    IndMin_init = int(np.argmin(r))

    # ── Vectorised over all time steps ───────────────────────────────────
    # vd shape: (tLen, N, 2)
    vd        = (v[None, :, None] * tVec[:, None, None]) * dirVec[None, :, :]
    NewPos    = PosUAV[None, :, :] + vd                   # (tLen, N, 2)
    NewRanges = np.sqrt(np.sum(NewPos**2, axis=2))        # (tLen, N)
    IndMins   = np.argmin(NewRanges, axis=1)              # (tLen,)

    HandoverMat[i, :] = (IndMins != IndMin_init).astype(np.uint8)

elapsed = time.time() - t_start
print(f"Done in {elapsed:.1f} s")

Handover_Uniform40DBS_Theory = HandoverMat.mean(axis=0)

os.makedirs("Data/Mobile_DBS", exist_ok=True)
np.save("Data/Mobile_DBS/Handover_Uniform40VelocityDBS_Theory.npy",
        Handover_Uniform40DBS_Theory)
print("Saved → Data/Mobile_DBS/Handover_Uniform40VelocityDBS_Theory.npy")
