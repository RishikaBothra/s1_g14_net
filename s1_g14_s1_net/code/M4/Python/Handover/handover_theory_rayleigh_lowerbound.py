"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   handover_theory_rayleigh_lowerbound.py                            ###
###   Purpose: Compute the lower-bound handover probability for DSM     ###
###            with Rayleigh-distributed DBS speed.                     ###
###            Corresponds to "DSM, Theory (Lower Bound)" in Fig. 2.    ###
###                                                                     ###
###   Randomised algorithm:                                             ###
###       • lambda0UAV ~ Uniform(5e-7, 2e-6)                            ###
###       • vMean_ms   ~ Uniform(30, 60) km/h → vSigma derived         ###
###       • T          ~ Uniform(200, 400) s                            ###
###       All drawn via randomised_config.sample_params()               ###
###########################################################################
"""

import sys, os, time
import numpy as np
from scipy.stats import rayleigh as rayleigh_dist
from scipy.integrate import quad, dblquad, tplquad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import load_params

# ── 1. Load the SAME parameters as the simulation script ──────────────────
params     = load_params("Data/handover_run_params.json", verbose=True)
lambda0UAV = params["lambda0UAV"]
vMean_ms   = params["vMean_ms"]
vSigma     = params["vSigma"]          # Rayleigh σ for speed distribution
T          = params["T"]
dt         = params["dt"]
tVec       = params["tVec_handover"]
tLen       = len(tVec)

u0Sigma = 1.0 / np.sqrt(2 * np.pi * lambda0UAV)
u0Mean  = u0Sigma * np.sqrt(np.pi / 2)

# Speed and nearest-neighbour distributions (vSigma randomised)
fv  = lambda v1: rayleigh_dist.pdf(v1, scale=vSigma)
Fv  = lambda v1: rayleigh_dist.cdf(v1, scale=vSigma)
fu0 = lambda u1: rayleigh_dist.pdf(u1, scale=u0Sigma)


def Fun1(v0, u0, theta0, t):
    """
    Inner integral I(v0, u0, theta0, t).
    Matches MATLAB Fun1 in the Rayleigh theory script.
    """
    R = np.sqrt(u0**2 + v0**2 * t**2 - 2 * u0 * v0 * t * np.cos(theta0))

    def q1_integrand(ux):
        arg = (u0 - ux) / t
        return ux * (1 - Fv(arg))

    def q2_integrand(ux, vi):
        arg = (vi**2 * t**2 + ux**2 - u0**2) / (2 * vi * t * ux)
        arg = np.clip(arg, -1.0, 1.0)
        return ux * fv(vi) / np.pi * np.arccos(arg)

    if R <= 0:
        return 0.0

    q1, _ = quad(q1_integrand, 0, R, limit=80, epsabs=1e-6, epsrel=1e-5)

    try:
        q2, _ = dblquad(
            q2_integrand,
            0, R,                            # ux limits
            lambda ux: abs(u0 - ux) / t,    # vi lower
            lambda ux: (u0 + ux) / t,       # vi upper
            epsabs=1e-6, epsrel=1e-5,
        )
    except Exception:
        q2 = 0.0

    return q1 - q2


# ── 2. Main loop over time ────────────────────────────────────────────────
Handover_RayleighDBS_Theory = np.zeros(tLen)

v_upper  = 3 * vMean_ms
u0_upper = 3 * u0Mean

for it, t in enumerate(tVec):
    t0 = time.time()
    print(f"t = {t:4d} s …", end=" ", flush=True)

    def integrand(theta0, u0_val, v0_val):
        inner = Fun1(v0_val, u0_val, theta0, t)
        return (np.exp(-2 * np.pi * lambda0UAV * inner)
                * fv(v0_val) * fu0(u0_val) / (2 * np.pi))

    # tplquad: innermost(z), then y limits, then x limits
    # variables: x=v0, y=u0, z=theta0
    q, _ = tplquad(
        integrand,
        0, v_upper,                   # v0 limits (outer)
        0, u0_upper,                  # u0 limits (middle)
        0, 2 * np.pi,                 # theta0 limits (inner)
        epsabs=1e-4, epsrel=1e-4,
    )

    Handover_RayleighDBS_Theory[it] = 1 - q
    print(f"P_H ≥ {Handover_RayleighDBS_Theory[it]:.4f}  [{time.time()-t0:.1f}s]")

# ── 3. Save ───────────────────────────────────────────────────────────────
os.makedirs("Data/Mobile_DBS", exist_ok=True)
np.save("Data/Mobile_DBS/Handover_RayleighVelocityDBS_Theory.npy",
        Handover_RayleighDBS_Theory)
print("\nSaved → Data/Mobile_DBS/Handover_RayleighVelocityDBS_Theory.npy")
