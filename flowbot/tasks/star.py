"""
Task: Star polygon in the XY plane.

A star is built by alternating outer tips and inner notches:
  - N_POINTS outer vertices at radius R_OUTER
  - N_POINTS inner vertices at radius R_INNER, halfway between outer tips

Default: 5-pointed star.
"""
import numpy as np

TASK_NAME    = "star"
N_TIPS       = 5       # number of star points
R_OUTER_MM   = 30.0   # radius of outer tips (mm)
R_INNER_MM   = 10.0    # radius of inner notches (mm)
Z_OFFSET     = 10.0    # height above home (mm)
HOLDING_TIME = 1    # hold time at each waypoint (s)


def _star_xy(n, r_outer, r_inner):
    """Return (x, y) arrays for a star polygon with 2n vertices."""
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n, endpoint=False)
    radii  = np.where(np.arange(2 * n) % 2 == 0, r_outer, r_inner)
    return radii * np.cos(angles), radii * np.sin(angles)


def get_waypoints(robot=None):
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z = 95.0 + Z_OFFSET

    x, y = _star_xy(N_TIPS, R_OUTER_MM, R_INNER_MM)
    # close the star by returning to first vertex
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    waypoints = []
    for xi, yi in zip(x, y):
        pc = np.array([xi, yi, z], dtype=float)
        waypoints.append((pc, HOLDING_TIME))
    return waypoints


def draw_reference(axes, robot=None):
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z = 95.0 + Z_OFFSET

    x, y = _star_xy(N_TIPS, R_OUTER_MM, R_INNER_MM)
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")

    axes["xy"].plot(x, y, **kw)
    axes["xz"].plot(x, np.full_like(x, z), **kw)
    axes["yz"].plot(y, np.full_like(y, z), **kw)
