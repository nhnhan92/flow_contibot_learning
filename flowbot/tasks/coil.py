"""
Task: Planar coil (Archimedean spiral) in the XY plane at constant Z.

Parametric form:
    r(t)  = R_MAX - (R_MAX - R_MIN) * t / (N_TURNS * 2π)
    x(t)  = r(t) * cos(t)
    y(t)  = r(t) * sin(t)
    z     = z0   (constant)

The robot starts at the outer radius, spirals inward, then reverses back
outward along the same spiral path.
"""
import numpy as np

TASK_NAME    = "coil"
R_MIN_MM     = 10.0   # inner radius (mm)
R_MAX_MM     = 28.0   # outer radius (mm)
N_TURNS      = 3      # number of full turns
Z_OFFSET     = 15.0    # height above home (mm)
N_POINTS     = 50     # total waypoints
HOLDING_TIME = 0.8      # hold time at each waypoint (s)


def _coil_xy(r_min, r_max, n_turns, n_pts):
    """Return (x, y) arrays for the planar Archimedean spiral."""
    t_max = n_turns * 2 * np.pi
    t = np.linspace(0, t_max, n_pts, endpoint=False)
    r = r_max - (r_max - r_min) * t / t_max
    return r * np.cos(t), r * np.sin(t)


def get_waypoints(robot=None, reverse=False):
    if robot is not None:
        z0 = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z0 = 95.0 + Z_OFFSET

    x, y = _coil_xy(R_MIN_MM, R_MAX_MM, N_TURNS, N_POINTS)

    if reverse:
        # Forward (outer → inner) then reverse (inner → outer), skip duplicate endpoint
        x = np.concatenate([x, x[-2::-1]])
        y = np.concatenate([y, y[-2::-1]])  

    waypoints = []
    for xi, yi in zip(x, y):
        pc = np.array([xi, yi, z0], dtype=float)
        waypoints.append((pc, HOLDING_TIME))
    return waypoints


def draw_reference(axes, robot=None):
    if robot is not None:
        z0 = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z0 = 95.0 + Z_OFFSET

    x, y = _coil_xy(R_MIN_MM, R_MAX_MM, N_TURNS, 400)

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")

    axes["xy"].plot(x, y, **kw)
    axes["xz"].plot(x, np.full_like(x, z0), **kw)
    axes["yz"].plot(y, np.full_like(y, z0), **kw)
