"""
Task: Circle in the XY plane at a fixed height.
"""
import numpy as np

TASK_NAME = "circle_xy"
SAMPLED_POINTS = 25   # number of waypoints to sample on the circle
RADIUS = 25.0   # circle radius in mm
Z_OFFSET = 10.0  # height of circle above home position (mm)
HOLDING_TIME = 0.75  # hold time at each waypoint (s)
def get_waypoints(robot=None, reverse=False):
    """
    Returns waypoints tracing a horizontal circle.
    robot: Flow_driven_bellow instance (used to get l0, lu for home height).
    """
    n_points  = SAMPLED_POINTS        # number of waypoints on the circle
    radius_mm = RADIUS         # circle radius in mm
    hold_s    = HOLDING_TIME       # hold time at each waypoint (s)
    z = Z_OFFSET                   # relative height of circle (mm above home)
    # Height: use home z if robot is available, else 95 mm
    if robot is not None:
        init_z = robot.flowbot.l0 + robot.flowbot.lu  # straight-up home height (mm)
    else:
        init_z = 95.0
    z += init_z                # relative height of circle (mm above home)

    fwd = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # forward then backward (skip duplicate start point)
    angles = np.concatenate([fwd, fwd[-2::-1]])
    if reverse:
        angles = angles[::-1]

    waypoints = []
    for a in angles:
        pc = np.array([radius_mm * np.cos(a),
                       radius_mm * np.sin(a),
                       z], dtype=float)
        waypoints.append((pc, hold_s))
    return waypoints


def draw_reference(axes, robot=None):
    """
    Draw the expected circle trajectory on all 2D projection plots.
      XY  : full circle
      XZ  : horizontal line  (z = circle_z, x in [-radius, +radius])
      YZ  : horizontal line  (z = circle_z, y in [-radius, +radius])
    axes: dict with keys "xy", "xz", "yz"  (from plot_helper.setup_plot)
    """
    radius_mm = RADIUS
    z_offset  = Z_OFFSET
    if robot is not None:
        z = robot.flowbot.l0 + robot.flowbot.lu + z_offset
    else:
        z = 95.0 + z_offset

    theta = np.linspace(0, 2 * np.pi, 200)
    cx = radius_mm * np.cos(theta)
    cy = radius_mm * np.sin(theta)

    kw = dict(color="green", linewidth=1.2, linestyle="--", alpha=0.7, label="ref")

    # XY: circle
    axes["xy"].plot(cx, cy, **kw)

    # XZ: horizontal line at z, x in [-r, +r]
    axes["xz"].plot([-radius_mm, radius_mm], [z, z], **kw)

    # YZ: horizontal line at z, y in [-r, +r]
    axes["yz"].plot([-radius_mm, radius_mm], [z, z], **kw)
