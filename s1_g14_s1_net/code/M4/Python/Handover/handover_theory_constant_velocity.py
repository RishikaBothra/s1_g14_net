"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                   ###
###                                                                     ###
###   handover_theory_constant_velocity.py                              ###
###   Purpose: Compute exact handover probability for SSM               ###
###            (constant / deterministic DBS speed).                    ###
###            Corresponds to "SSM, Theory (Exact)" in Fig. 2.          ###
###                                                                     ###
###   Randomised algorithm:                                             ###
###       • lambda0UAV ~ Uniform(5e-7, 2e-6)  [UAVs/m²]               ###
###       • vMean_ms   ~ Uniform(30, 60) km/h → used as constant v     ###
###       • T          ~ Uniform(200, 400) s                            ###
###       All drawn via randomised_config.sample_params()               ###
###########################################################################
"""

import sys, os, time
import numpy as np
from scipy.stats import rayleigh as rayleigh_dist
from scipy.integrate import quad, dblquad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomised_config import load_params

# ── 1. Load the SAME parameters as the simulation script ──────────────────
params     = load_params("Data/handover_run_params.json", verbose=True)
lambda0UAV = params["lambda0UAV"]
v          = params["vMean_ms"]          # constant speed for SSM
T          = params["T"]
dt         = params["dt"]
tVec       = params["tVec_handover"]
tLen       = len(tVec)

# ── Nearest-neighbour distance distribution (Rayleigh with σ = u0Sigma) ──
u0Sigma = 1.0 / np.sqrt(2 * np.pi * lambda0UAV)   # derived from λ₀
fu0     = lambda u1: rayleigh_dist.pdf(u1, scale=u0Sigma)

# ── Auxiliary area integral for a given (u0, theta0) ─────────────────────
def Fun1(u0, theta0, v, t):
    """∫ ux/π · arccos(…) dux  from |v·t - u0| to R"""
    R = np.sqrt(u0**2 + v**2 * t**2 - 2 * u0 * v * t * np.cos(theta0))
    lo = abs(v * t - u0)
    if R <= lo:
        return 0.0

    def integrand(ux):
        arg = (u0**2 - ux**2 - v**2 * t**2) / (2 * ux * v * t)
        arg = np.clip(arg, -1.0, 1.0)
        return ux / np.pi * np.arccos(arg)

    I, _ = quad(integrand, lo, R, limit=100, epsabs=1e-8, epsrel=1e-6)
    return I

# ── 2. Main loop over time ────────────────────────────────────────────────
Handover_ConstantDBS_Theory = np.zeros(tLen)

for it, t in enumerate(tVec):
    t0 = time.time()
    print(f"t = {t:4d} s …", end=" ", flush=True)

    # ── Integrand over region u0 ∈ [0, v·t] (UAV closer than displacement) ─
    # Default-arg capture avoids Python late-binding closure bug with loop var t
    def fun01(u0_val, theta0_val, _t=t, _v=v):
        inner = Fun1(u0_val, theta0_val, _v, _t)
        return (np.exp(-2 * np.pi * lambda0UAV * inner)
                * np.exp(-np.pi * lambda0UAV * (_v * _t - u0_val)**2)
                * fu0(u0_val) / (2 * np.pi))

    # ── Integrand over region u0 ∈ [v·t, ∞) ────────────────────────────
    def fun02(u0_val, theta0_val, _t=t, _v=v):
        inner = Fun1(u0_val, theta0_val, _v, _t)
        return (np.exp(-2 * np.pi * lambda0UAV * inner)
                * fu0(u0_val) / (2 * np.pi))

    # Truncated upper limit matching MATLAB (1e4 m)
    u0_upper = 1e4

    # dblquad signature: integrand(y, x) where x is outer var
    q1, _ = dblquad(
        lambda theta0, u0_v: fun01(u0_v, theta0),
        0, v * t,                     # u0 limits
        0, 2 * np.pi,                 # theta0 limits
        epsabs=1e-5, epsrel=1e-4,
    )
    q2, _ = dblquad(
        lambda theta0, u0_v: fun02(u0_v, theta0),
        v * t, u0_upper,
        0, 2 * np.pi,
        epsabs=1e-5, epsrel=1e-4,
    )

    q = q1 + q2
    Handover_ConstantDBS_Theory[it] = 1 - q
    print(f"P_H = {Handover_ConstantDBS_Theory[it]:.4f}  [{time.time()-t0:.1f}s]")

# ── 3. Save ───────────────────────────────────────────────────────────────
os.makedirs("Data/Mobile_DBS", exist_ok=True)
np.save("Data/Mobile_DBS/Handover_ConstantVelocityDBS_Theory.npy",
        Handover_ConstantDBS_Theory)
print("\nSaved → Data/Mobile_DBS/Handover_ConstantVelocityDBS_Theory.npy")
