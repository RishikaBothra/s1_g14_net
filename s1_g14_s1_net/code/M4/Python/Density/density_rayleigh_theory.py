"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   density_rayleigh_theory.py                                        ###
###   Purpose: Compute the theoretical spatial density of non-serving   ###
###            DBSs (DSM, Rayleigh-distributed speed).                  ###
###            Corresponds to Fig. 1 theory curves.                     ###
###                                                                     ###
###   Randomised algorithm: lambda0UAV, vSigma, u0, and tVec are       ###
###   drawn from distributions via randomised_config.sample_params().   ###
###########################################################################
"""

import sys, os, time
import numpy as np
from scipy.stats import rayleigh as rayleigh_dist
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import sample_params, save_params

# ── 1. Sample randomised parameters ──────────────────────────────────────
params = sample_params(verbose=True)

lambda0UAV = params["lambda0UAV"]
R_UAV      = params["R_UAV_density"]
vSigma     = params["vSigma"]
u0         = params["u0"]
tVec       = params["tVec_density"]
dr         = params["dr"]

# Save so the simulation script uses the IDENTICAL parameter draw
os.makedirs("Data", exist_ok=True)
save_params(params, "Data/density_run_params.json")

NumR = int(round((R_UAV - dr) / dr)) + 1
tLen = len(tVec)

# Rayleigh speed PDF and CDF (vSigma is randomised)
fv = lambda v1: rayleigh_dist.pdf(v1, scale=vSigma)
Fv = lambda v1: rayleigh_dist.cdf(v1, scale=vSigma)

Density_Theory = np.zeros((tLen, NumR))
os.makedirs("Data", exist_ok=True)

for kk, t in enumerate(tVec):
    t_start = time.time()
    print(f"Processing t = {t} s …")

    ux_values = np.arange(dr, round(R_UAV) + 1, dr)

    for ux in ux_values:
        idx = int(round(ux / dr)) - 1
        if idx >= NumR:
            break

        # ── Term A: P(v > |u0 - ux| / t) ───────────────────────────────
        v_lower_A = max(0.0, (u0 - ux) / t)
        A = 1.0 - Fv(v_lower_A)

        # ── Term B: integral of fv * (1/π) * arccos(…) ──────────────────
        v_lower_B = abs(u0 - ux) / t
        v_upper_B = (u0 + ux) / t

        # NOTE: default-argument capture (t=t, ux=ux, u0=u0) prevents the
        # classic Python late-binding closure bug inside a loop.
        def integrand_B(vi, _t=t, _ux=ux, _u0=u0):
            arg = (vi**2 * _t**2 + _ux**2 - _u0**2) / (2.0 * vi * _t * _ux)
            arg = np.clip(arg, -1.0, 1.0)
            return fv(vi) * (1.0 / np.pi) * np.arccos(arg)

        if v_upper_B > v_lower_B:
            B, _ = quad(integrand_B, v_lower_B, v_upper_B,
                        limit=200, epsabs=1e-10, epsrel=1e-8)
        else:
            B = 0.0

        Density_Theory[kk, idx] = lambda0UAV * (A - B)

    elapsed = time.time() - t_start
    print(f"  t = {t} s done in {elapsed:.1f} s")

# ── 3. Save ───────────────────────────────────────────────────────────────
np.save("Data/Density_RayleighVelocity_Theory.npy", Density_Theory)
print("Saved → Data/Density_RayleighVelocity_Theory.npy")
print(f"Density_Theory shape: {Density_Theory.shape}")
