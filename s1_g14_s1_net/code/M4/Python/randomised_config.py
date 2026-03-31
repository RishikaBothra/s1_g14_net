"""
###########################################################################
###                                                                     ###
###   Paper: Handover Probability in Drone Cellular Networks            ###
###   Authors: Morteza Banagar, Vishnu V. Chetlur,                      ###
###            and Harpreet S. Dhillon                                  ###
###                                                                     ###
###   randomised_config.py                                              ###
###   Purpose: Centralised RANDOMISED parameter generator for all       ###
###            simulation scripts.  Instead of hard-coded values the    ###
###            key physical parameters are drawn from well-motivated    ###
###            probability distributions each time this module is       ###
###            imported (or `sample_params()` is called).               ###
###                                                                     ###
###   Randomised variables                                              ###
###   ─────────────────────                                             ###
###   lambda0UAV : UAV density  ~ Uniform(5e-7, 2e-6)  [UAVs/m²]        ###
###   vMean_kmh  : mean speed   ~ Uniform(30, 60)       [km/h]          ###
###   vDiff_kmh  : speed range  ~ Uniform(20, 120)      [km/h]          ###
###                (width of uniform speed distribution)                ###
###   u0         : exclusion radius ~ Uniform(300, 800) [m]             ###
###   T          : simulation window ~ Uniform(200, 400) [s] (int)      ###
###                                                                     ###
###   Fixed / derived values (same as MATLAB originals)                 ###
###   ─────────────────────────────────────────────────                 ###
###   R_UAV  = 1e4 m  (density fig)  / 1e5 m (handover fig)             ###
###   dr     = 1 m                                                      ###
###   dt     = 1 s                                                      ###
###########################################################################
"""

import numpy as np


# ---------------------------------------------------------------------------
# Seed control (set GLOBAL_SEED = None for fully random runs)
# ---------------------------------------------------------------------------
GLOBAL_SEED: int | None = None   # Change to an integer for reproducibility


def _rng(seed=None) -> np.random.Generator:
    """Return a fresh numpy Generator."""
    return np.random.default_rng(seed)


def sample_params(seed=None, verbose: bool = True) -> dict:
    """
    Draw all randomised simulation parameters from their distributions.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.  If None uses GLOBAL_SEED.
    verbose : bool
        Print sampled parameters to stdout.

    Returns
    -------
    dict with keys:SSS
        lambda0UAV, R_UAV_density, R_UAV_handover,
        vMean_ms, vSigma, vStart, vEnd,
        u0, T, dt, dr, tVec_density, tVec_handover
    """
    rng = _rng(seed if seed is not None else GLOBAL_SEED)

    # ── Core randomised parameters ────────────────────────────────────────
    lambda0UAV: float = rng.uniform(5e-7, 2e-6)          # UAV spatial density
    vMean_kmh:  float = rng.uniform(30.0, 60.0)           # mean speed [km/h]
    vDiff_kmh:  float = rng.uniform(20.0, 120.0)          # speed spread [km/h]
    u0:         float = rng.uniform(300.0, 800.0)         # exclusion radius [m]
    T:          int   = int(rng.uniform(200, 400))        # time window [s]

    # ── Derived quantities (deterministic from sampled values) ────────────
    vMean_ms: float = vMean_kmh / 3.6                     # [m/s]
    vSigma:   float = vMean_ms * np.sqrt(2 / np.pi)      # Rayleigh σ
    vDiff_ms: float = vDiff_kmh / 3.6
    vStart:   float = max(0.0, vMean_ms - vDiff_ms / 2)  # uniform lower bound
    vEnd:     float = vMean_ms + vDiff_ms / 2             # uniform upper bound

    # ── Fixed geometry / time resolution ─────────────────────────────────
    R_UAV_density:  float = 1e4   # [m]  – density figure
    R_UAV_handover: float = 1e5   # [m]  – handover figure
    dr: float = 1.0               # radial bin width [m]
    dt: int   = 1                 # time step [s]

    # t-vectors
    tVec_density:  list = [10, 20, 40, 100]               # density figure
    tVec_handover: list = list(range(dt, T + 1, dt))      # handover figure

    params = dict(
        lambda0UAV     = lambda0UAV,
        R_UAV_density  = R_UAV_density,
        R_UAV_handover = R_UAV_handover,
        vMean_ms       = vMean_ms,
        vSigma         = vSigma,
        vStart         = vStart,
        vEnd           = vEnd,
        u0             = u0,
        T              = T,
        dt             = dt,
        dr             = dr,
        tVec_density   = tVec_density,
        tVec_handover  = tVec_handover,
    )

    if verbose:
        print("=" * 60)
        print("  RANDOMISED SIMULATION PARAMETERS")
        print("=" * 60)
        print(f"  lambda0UAV  = {lambda0UAV:.3e}  [UAVs/m²]")
        print(f"  vMean       = {vMean_kmh:.2f} km/h  ({vMean_ms:.4f} m/s)")
        print(f"  vDiff       = {vDiff_kmh:.2f} km/h")
        print(f"  vStart      = {vStart:.4f} m/s")
        print(f"  vEnd        = {vEnd:.4f} m/s")
        print(f"  vSigma      = {vSigma:.4f} m/s")
        print(f"  u0          = {u0:.2f} m")
        print(f"  T           = {T} s")
        print(f"  tVec (den.) = {tVec_density}")
        print("=" * 60)

    return params


# ---------------------------------------------------------------------------
# Parameter persistence – makes theory + simulation share one parameter set
# ---------------------------------------------------------------------------
import json, os

def save_params(params: dict, path: str) -> None:
    """
    Persist params dict to a JSON file so other scripts can reload the
    SAME randomised draw without resampling.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serialisable = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in params.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[params] Saved → {path}")


def load_params(path: str, verbose: bool = True) -> dict:
    """
    Load a previously saved params dict from a JSON file.
    Falls back to sampling new params if the file does not exist.
    """
    if not os.path.exists(path):
        print(f"[params] WARNING: '{path}' not found — sampling new parameters.")
        return sample_params(verbose=verbose)

    with open(path) as f:
        params = json.load(f)

    params["T"] = int(params["T"])   # restore integer type

    if verbose:
        print("=" * 60)
        print(f"  LOADED PARAMETERS from {os.path.basename(path)}")
        print("=" * 60)
        print(f"  lambda0UAV = {params['lambda0UAV']:.3e}  [UAVs/m²]")
        print(f"  vMean_ms   = {params['vMean_ms']:.4f} m/s")
        print(f"  vSigma     = {params['vSigma']:.4f} m/s")
        print(f"  u0         = {params['u0']:.2f} m")
        print(f"  T          = {params['T']} s")
        print("=" * 60)

    return params


if __name__ == "__main__":
    p = sample_params()
