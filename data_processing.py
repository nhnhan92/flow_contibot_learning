"""
data_processing.py — Fit kinematic calibration models from an elongation CSV.

CSV format (produced by data_logging.py elong command):
    t_s, pwm_cmd, proc_flow1, proc_flow2, proc_flow3, proc_press,
    opti_x_mm, opti_y_mm, opti_z_mm

Models fitted:
    pwm2flow  : flow [L/min] = a*pwm + b
    flow2press: pressure [MPa] = a*flow + b
    pwm2press : pressure [MPa] = a*pwm + b  (direct fit, forward model)
    stiffness : k [N/mm] = k0 + a*(1 - exp(-b*delta_l))
                where delta_l = opti_z_mm (extension length)

Usage:
    python data_processing.py                          # uses FILE_PATH below
    python data_processing.py path/to/file.csv        # override from CLI
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Config ────────────────────────────────────────────────────────────────────
FILE_PATH  = "data/elongation_calib/log_elongation_calib_20260706_221343_elongation.csv"

D_in       = 5.0    # mm — inner diameter of actuator cross-section
D_out      = 16.5   # mm — outer diameter
Aeff       = np.pi * (D_in + D_out) ** 2 / 16   # effective area (mm²)

SKIP_FRAC  = 0.4    # fraction of each constant-PWM segment to skip (transients)
MIN_EXT_MM = 0.5    # ignore stiffness points where extension is near zero

# ── Step 1: Load data ─────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    FILE_PATH = sys.argv[1]

df = pd.read_csv(FILE_PATH)
print(f"Loaded {len(df)} rows from {FILE_PATH}")
print(f"PWM levels: {sorted(df['pwm_cmd'].unique())}")

# ── Step 2: Steady-state segment averaging ────────────────────────────────────
def steady_state_means(df: pd.DataFrame, skip_frac: float = 0.4) -> pd.DataFrame:
    """
    Group consecutive rows with the same pwm_cmd into segments.
    Skip the first skip_frac rows of each segment (transient settling),
    then average the remainder.
    """
    seg_id = (df["pwm_cmd"].diff() != 0).cumsum()
    rows = []
    for _, seg in df.groupby(seg_id):
        pwm = seg["pwm_cmd"].iloc[0]
        skip = int(len(seg) * skip_frac)
        steady = seg.iloc[skip:]
        if len(steady) == 0:
            continue
        rows.append({
            "pwm_cmd":    pwm,
            "flow1":      steady["proc_flow1"].mean(),
            "flow2":      steady["proc_flow2"].mean(),
            "flow3":      steady["proc_flow3"].mean(),
            "flow_avg":   steady[["proc_flow1", "proc_flow2", "proc_flow3"]].values.mean(),
            "press":      steady["proc_press"].mean(),
            "ext_mm":     steady["opti_z_mm"].mean(),
        })
    return pd.DataFrame(rows)

ss = steady_state_means(df, SKIP_FRAC)

# Drop rest state (pwm=0) before fitting models
ss_fit = ss[ss["pwm_cmd"] > 0].copy()
print(f"\nSteady-state segments (pwm>0): {len(ss_fit)}")
print(ss_fit[["pwm_cmd", "flow_avg", "press", "ext_mm"]].to_string(index=False))

# ── Step 3: Fit models ────────────────────────────────────────────────────────

# --- 3a. pwm2flow: flow [L/min] = a*pwm + b ---
pwm_v  = ss_fit["pwm_cmd"].values.astype(float)
flow_v = ss_fit["flow_avg"].values

p2f = np.polyfit(pwm_v, flow_v, 1)
print(f"\n[pwm2flow]  flow = {p2f[0]:.5f}*pwm + {p2f[1]:.5f}  (L/min)")

# --- 3b. flow2press: pressure [MPa] = a*flow + b ---
press_v = ss_fit["press"].values

f2p = np.polyfit(flow_v, press_v, 1)
print(f"[flow2press] press = {f2p[0]:.6f}*flow + {f2p[1]:.6f}  (MPa)")

# --- 3c. pwm2press: press = a*pwm + b  (direct forward fit) ---
p2p = np.polyfit(pwm_v, press_v, 1)
print(f"[pwm2press]  press = {p2p[0]:.6f}*pwm + {p2p[1]:.6f}  (MPa)")

# --- 3d. stiffness: k [N/mm] = k0 + a*(1 - exp(-b*delta_l)) ---
ext_v = ss_fit["ext_mm"].values
valid = ext_v > MIN_EXT_MM
ext_s   = ext_v[valid]
press_s = press_v[valid]

k_data = press_s * Aeff / ext_s   # N/mm

def sat_exp(x, k0, a, b):
    return k0 + a * (1.0 - np.exp(-b * x))

try:
    p0 = [k_data.min(), k_data.max() - k_data.min(), 0.1]
    popt, _ = curve_fit(sat_exp, ext_s, k_data, p0=p0, maxfev=10000)
    k0, a_k, b_k = popt
    print(f"[stiffness]  k = {k0:.5f} + {a_k:.5f}*(1 - exp(-{b_k:.5f}*delta_l))  (N/mm)")
    stiffness_ok = True
except RuntimeError:
    print("[stiffness]  curve_fit failed — check extension data")
    stiffness_ok = False

# ── Step 4: Print Python-ready lambdas ────────────────────────────────────────
print("\n" + "=" * 60)
print("# Copy-paste into your kinematic model:")
print(f"pwm2flow_model   = lambda pwm:     {p2f[0]:.5f}*pwm + {p2f[1]:.5f}")
print(f"flow2press_model = lambda flow:    {f2p[0]:.6f}*flow + {f2p[1]:.6f}")
print(f"pwm2press_model  = lambda pwm:     {p2p[0]:.6f}*pwm + {p2p[1]:.6f}")
if stiffness_ok:
    print(f"stiffness_model  = lambda delta_l: {k0:.5f} + {a_k:.5f}*(1 - np.exp(-{b_k:.5f}*delta_l))")
print("=" * 60)

# ── Step 5: Plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(Path(FILE_PATH).stem)
axes = axes.flatten()

# pwm → flow
ax = axes[0]
ax.scatter(pwm_v, flow_v, label="data")
pwm_line = np.linspace(pwm_v.min(), pwm_v.max(), 100)
ax.plot(pwm_line, np.polyval(p2f, pwm_line), "r-", label="fit")
ax.set_xlabel("PWM command"); ax.set_ylabel("Flow avg (L/min)")
ax.set_title("pwm2flow"); ax.legend()

# flow → pressure
ax = axes[1]
ax.scatter(flow_v, press_v, label="data")
flow_line = np.linspace(flow_v.min(), flow_v.max(), 100)
ax.plot(flow_line, np.polyval(f2p, flow_line), "r-", label="fit")
ax.set_xlabel("Flow avg (L/min)"); ax.set_ylabel("Pressure (MPa)")
ax.set_title("flow2press"); ax.legend()

# pwm → pressure (direct forward)
ax = axes[2]
ax.scatter(pwm_v, press_v, label="data")
ax.plot(pwm_line, np.polyval(p2p, pwm_line), "r-", label="fit")
ax.set_xlabel("PWM command"); ax.set_ylabel("Pressure (MPa)")
ax.set_title("pwm2press"); ax.legend()

# extension → stiffness
ax = axes[3]
if stiffness_ok and len(ext_s) > 0:
    ax.scatter(ext_s, k_data, label="data")
    ext_line = np.linspace(ext_s.min(), ext_s.max(), 100)
    ax.plot(ext_line, sat_exp(ext_line, k0, a_k, b_k), "r-", label="fit")
    ax.set_xlabel("Extension delta_l (mm)"); ax.set_ylabel("Stiffness k (N/mm)")
    ax.set_title("stiffness"); ax.legend()
else:
    ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()

out_path = Path(FILE_PATH).with_suffix(".png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
