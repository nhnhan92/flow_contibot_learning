"""
Task: Wavy circle (rhodonea / undulating circle) in the XY plane.

The radius oscillates sinusoidally as the angle sweeps around:
    r(θ) = R + A * sin(N_WAVES * θ)

N_WAVES = 4  → 4 outward bumps and 4 inward dips per revolution.
"""
import numpy as np

TASK_NAME    = "wavy_circle"
RADIUS_MM    = 15.0   # mean radius (mm)
AMPLITUDE_MM = 4.0    # peak radial deviation (mm)
N_WAVES      = 5      # number of oscillations per full circle
Z_OFFSET     = 5.0    # height above home (mm)
N_POINTS     = 60     # waypoints per full loop
HOLDING_TIME = 0.7    # hold time at each waypoint (s)


def _wavy_circle_xy(R, A, n_waves, n_pts):
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r = R + A * np.sin(n_waves * theta)
    return r * np.cos(theta), r * np.sin(theta)


def get_waypoints(robot=None):
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + Z_OFFSET
    else:
        z = 95.0 + Z_OFFSET

    x, y = _wavy_circle_xy(RADIUS_MM, AMPLITUDE_MM, N_WAVES, N_POINTS)

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

    x, y = _wavy_circle_xy(RADIUS_MM, AMPLITUDE_MM, N_WAVES, 400)
    # close the curve for plotting
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")

    axes["xy"].plot(x, y, **kw)
    axes["xz"].plot(x, np.full_like(x, z), **kw)
    axes["yz"].plot(y, np.full_like(y, z), **kw)
