"""
Task: Bernoulli lemniscate (figure-8 / infinity symbol) in the XY plane.

Parametric form:
    x(t) = a * sqrt(2) * cos(t) / (1 + sin²(t))
    y(t) = a * sqrt(2) * sin(t)*cos(t) / (1 + sin²(t))
  t ∈ [0, 2π] traces the full figure-8 once.

ROTATIONS controls which variants to run (executed in order):
    ROTATIONS = [0.0]         – horizontal only
    ROTATIONS = [90.0]        – vertical only
    ROTATIONS = [0.0, 90.0]   – both in one run (default)
"""
import numpy as np

TASK_NAME    = "lemniscate"
ROTATIONS    = [0.0]   # list of rotation angles (deg) to execute in sequence
SEMI_AXIS_MM = 25.0           # half-width of the figure-8 (mm)
Z_OFFSET     = 10.0           # height above home (mm)
N_POINTS     = 40             # waypoints per full loop
HOLDING_TIME = 0.75           # hold time at each waypoint (s)


def _lemniscate_xy(a, n, rotation_deg=0.0):
    """Return (x, y) arrays for one full lemniscate loop, rotated by rotation_deg."""
    t     = np.linspace(0, 2 * np.pi, n, endpoint=False)
    denom = 1 + np.sin(t) ** 2
    x     = a * np.sqrt(2) * np.cos(t) / denom
    y     = a * np.sqrt(2) * np.sin(t) * np.cos(t) / denom
    if rotation_deg != 0.0:
        c    = np.cos(np.radians(rotation_deg))
        s    = np.sin(np.radians(rotation_deg))
        x, y = c * x - s * y, s * x + c * y
    return x, y


def get_waypoints(robot=None, reverse=False):
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z = 95.0 + Z_OFFSET

    waypoints = []
    for rot in ROTATIONS:
        x, y = _lemniscate_xy(SEMI_AXIS_MM, N_POINTS, rotation_deg=rot)
        if reverse:
            x = np.concatenate([x, x[-2::-1]])
            y = np.concatenate([y, y[-2::-1]])
        for xi, yi in zip(x, y):
            waypoints.append((np.array([xi, yi, z], dtype=float), HOLDING_TIME))
    return waypoints


def draw_reference(axes, robot=None):
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z = 95.0 + Z_OFFSET

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")
    for i, rot in enumerate(ROTATIONS):
        x, y = _lemniscate_xy(SEMI_AXIS_MM, 300, rotation_deg=rot)
        kw_i = kw if i == 0 else {**kw, "label": "_nolegend_"}
        axes["xy"].plot(x, y, **kw_i)
        axes["xz"].plot(x, np.full_like(x, z), **kw_i)
        axes["yz"].plot(y, np.full_like(y, z), **kw_i)
