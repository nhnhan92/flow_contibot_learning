"""
Benchmark inference time for each stage of the kinematic model chain.

Run from program/ directory:
    python benchmark_ik.py
"""
from __future__ import annotations
import time, os, sys
from pathlib import Path
import numpy as np

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
FLOWBOT_DIR = Path(FILE_DIR) / "flowbot"
sys.path.insert(0, FILE_DIR)

N = 2000   # number of repetitions for timing

# ─────────────────────────────────────────────
# 1. Load models
# ─────────────────────────────────────────────
from flowbot.pwm2flow            import Pwm2FlowModel
from flowbot.pressure_flow_model import Flow2PressModel, Press2FlowModel
from flowbot.kinematic_modeling  import Flow_driven_bellow

t0 = time.perf_counter()
pwm2flow   = Pwm2FlowModel.load(  FLOWBOT_DIR / "pwm2flow.pkl")
flow2press = Flow2PressModel.load( FLOWBOT_DIR / "flow2press.pkl")
press2flow = Press2FlowModel.load( FLOWBOT_DIR / "press2flow.pkl")
load_ms = (time.perf_counter() - t0) * 1e3
print(f"\nModel loading:               {load_ms:.1f} ms  (one-time cost)")

robot = Flow_driven_bellow(
    D_in=5, D_out=16.5, l0=82, d=28.17, lb=0.0, lu=13.5,
    k_model=lambda dl: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * dl)),
    a_delta=0, b_delta=0,
    pwm2flow_model=pwm2flow, flow2press_model=flow2press, press2flow_model=press2flow,
)

# ─────────────────────────────────────────────
# helper
# ─────────────────────────────────────────────
def bench(name: str, fn, *args, n=N):
    # warm-up
    for _ in range(10):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"  {name:<45s} {ms*1e3:8.1f} us   ({ms:.4f} ms)")
    return ms

# representative inputs
pwm_in = np.array([5, 20, 5], dtype=float)
pb_in  = robot.pwm_to_pressure(pwm_in)
fk     = robot.forward_kinematics_from_pressures(pb_in)
pc_in  = fk["pc"]
l_in   = fk["l"]

flow_val  = pwm2flow.predict(15.0)
press_val = flow2press.predict(flow_val)

print(f"\n{'-'*65}")
print("Stage-by-stage timings  (avg over {:,} calls each)".format(N))
print(f"{'-'*65}")

# ─────────────────────────────────────────────
# 2. Individual model calls
# ─────────────────────────────────────────────
print("\n[Scalar models - 1 call]")
bench("pwm2flow.predict(pwm)",               pwm2flow.predict,   15.0)
bench("flow2press.predict(flow)",             flow2press.predict, flow_val)
bench("press2flow.predict(pressure)",         press2flow.predict, press_val)
bench("pwm2flow.predict_inverse(flow) [brentq]",
      pwm2flow.predict_inverse, flow_val)

# ─────────────────────────────────────────────
# 3. Kinematic model stages
# ─────────────────────────────────────────────
print("\n[Kinematic stages - per full robot call]")
bench("pwm_to_pressure(pwm×3)",              robot.pwm_to_pressure,  pwm_in)
bench("pressures_to_lengths(pb×3)",          robot.pressures_to_lengths, pb_in)
bench("lengths_to_config(l×3)",              robot.lengths_to_config, l_in)
bench("forward_kinematics_from_lengths(l×3)",robot.forward_kinematics_from_lengths, l_in)
bench("forward_kinematics_from_pressures(pb×3)", robot.forward_kinematics_from_pressures, pb_in)
bench("inverse_kinematics_position_to_lengths(pc)",
      robot.inverse_kinematics_position_to_lengths, pc_in)
bench("inverse_pressures_from_lengths(l×3)", robot.inverse_pressures_from_lengths, l_in)
bench("pressure_to_pwm(pb×3)  [3× brentq]", robot.pressure_to_pwm, pb_in)

# ─────────────────────────────────────────────
# 4. Full IK (what step() calls)
# ─────────────────────────────────────────────
print("\n[Full IK chain - what fb.step() calls every tick]")
t_ik = bench("inverse_pressures_from_position(pc)  TOTAL IK",
             robot.inverse_pressures_from_position, pc_in)

print(f"\n{'-'*65}")
print(f"  Full IK total: {t_ik:.4f} ms per call")
max_hz = 1.0 / (t_ik / 1e3)
print(f"  Max achievable rate (IK alone): {max_hz:.0f} Hz")
print(f"  Budget at 30 Hz: 33.3 ms   IK uses {t_ik/33.3*100:.1f}% of it")
print(f"{'-'*65}\n")
