"""
Task: Helical trajectory — circular motion in XY while rising in Z.

Parametric form:
    x(t) = R * cos(t)
    y(t) = R * sin(t)
    z(t) = z0 + PITCH_MM * t / (2π)   for t ∈ [0, N_TURNS * 2π]

The robot starts at the bottom of the helix and spirals upward.
"""
import numpy as np

TASK_NAME    = "helix"
RADIUS_MM    = 20.0   # radius of the helix (mm)
PITCH_MM     = 10.0    # vertical rise per full turn (mm)
N_TURNS      = 2      # number of full turns
Z_OFFSET     = 10.0    # height of helix bottom above home (mm)
N_POINTS     = 60     # total waypoints across all turns
HOLDING_TIME = 1    # hold time at each waypoint (s)


def _helix_xyz(R, pitch, n_turns, n_pts, z0):
    """Return (x, y, z) arrays for the helical path."""
    t = np.linspace(0, n_turns * 2 * np.pi, n_pts, endpoint=False)
    x = R * np.cos(t)
    y = R * np.sin(t)
    z = z0 + pitch * t / (2 * np.pi)
    return x, y, z


def get_waypoints(robot=None):
    if robot is not None:
        z0 = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z0 = 95.0 + Z_OFFSET

    x, y, z = _helix_xyz(RADIUS_MM, PITCH_MM, N_TURNS, N_POINTS, z0)

    waypoints = []
    for xi, yi, zi in zip(x, y, z):
        pc = np.array([xi, yi, zi], dtype=float)
        waypoints.append((pc, HOLDING_TIME))
    return waypoints


def draw_reference(axes, robot=None):
    if robot is not None:
        z0 = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z0 = 95.0 + Z_OFFSET

    x, y, z = _helix_xyz(RADIUS_MM, PITCH_MM, N_TURNS, 400, z0)

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")

    axes["xy"].plot(x, y, **kw)
    axes["xz"].plot(x, z, **kw)
    axes["yz"].plot(y, z, **kw)
