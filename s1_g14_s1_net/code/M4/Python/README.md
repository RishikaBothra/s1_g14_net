# Python – Handover Probability in Drone Cellular Networks
# Randomised Algorithm Implementation

> Python port of the MATLAB simulation codebase for the research paper
> **"Handover Probability in Drone Cellular Networks"** by Banagar, Chetlur & Dhillon.
> All key physical parameters are **randomised** (drawn from distributions) instead of fixed.

---

## File Structure

```
Python/
├── randomised_config.py          ← Shared parameter sampler (import this first!)
│
├── Density/
│   ├── density_rayleigh_simulation.py   ← Monte-Carlo density sim (Fig. 1)
│   ├── density_rayleigh_theory.py       ← Analytical density (Fig. 1)
│   └── plot_densities.py                ← Plot Fig. 1
│
├── Handover/
│   ├── handover_dbs_simulation.py           ← MC sim: SSM + DSM (Fig. 2)
│   ├── handover_theory_constant_velocity.py ← SSM exact theory (Fig. 2)
│   ├── handover_theory_rayleigh_lowerbound.py ← DSM Rayleigh lower bound (Fig. 2)
│   ├── handover_theory_uniform_lowerbound.py  ← DSM Uniform lower bound (Fig. 2)
│   └── plot_handover.py                     ← Plot Fig. 2
│
└── README.md
```

---

## Randomised Variables

| Variable | Distribution | Range | Unit |
|---|---|---|---|
| `lambda0UAV` | Uniform | [5×10⁻⁷, 2×10⁻⁶] | UAVs/m² |
| `vMean` | Uniform | [30, 60] | km/h |
| `vDiff` | Uniform | [20, 120] | km/h |
| `u0` | Uniform | [300, 800] | m (exclusion radius) |
| `T` | Uniform (int) | [200, 400] | s (time window) |

All derived quantities (`vSigma`, `vStart`, `vEnd`, `u0Sigma`) are computed deterministically from the above.

---

## How to Run

### Install dependencies
```bash
pip install numpy scipy matplotlib
```

### Fig. 1 – Density
```bash
cd Density
python density_rayleigh_theory.py      # may take ~5 min
python density_rayleigh_simulation.py  # ~2–5 min
python plot_densities.py
```

### Fig. 2 – Handover Probability
```bash
cd Handover
python handover_dbs_simulation.py                  # ~2–10 min
python handover_theory_constant_velocity.py        # slow (2-D integrals)
python handover_theory_rayleigh_lowerbound.py      # slow (3-D integrals)
python handover_theory_uniform_lowerbound.py       # MC, fast
python plot_handover.py
```

> **Tip**: Run the simulation scripts first; `plot_handover.py` gracefully skips any missing files.

---

## Why Randomised?

The deterministic / original MATLAB approach fixes:
- `lambda0UAV = 1e-6` (always)
- `vMean = 45 km/h` (always)
- `u0 = 500 m` (always)

With the **randomised algorithm**, each run samples these from distributions, making the simulation a proper stochastic experiment. This is useful for:
1. **Robustness analysis** – does the handover curve shape hold across parameter regimes?
2. **Sensitivity / Monte-Carlo sensitivity studies**
3. Comparing vs the deterministic result to show where the fixed-parameter assumption breaks down.
