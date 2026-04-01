"""
Task: Step-response test.
Drives the tip to a set of discrete positions (one axis at a time),
then returns to home. Good for measuring settling time and steady-state error.

All candidate waypoints are validated against the robot workspace hull before
being included; out-of-workspace points are silently skipped.
"""
import numpy as np

TASK_NAME        = "step_response"
Z_MIN_OFFSET     = 5.0    # min Z above home (mm)
Z_MAX_OFFSET     = 20.0   # max Z above home (mm)
N_POINTS_SAMPLED = 10     # number of XY amplitude steps
N_POINTS_Z       = 4      # number of Z levels
XY_MAX_MM        = 30.0   # maximum XY step amplitude (mm)
HOLD_S           = 1.0    # hold time at each step position (s)
HOLD_HOME_S      = 1.0    # hold time at home (s)


def _is_valid(pt, robot):
    """Return True if pt is inside the workspace hull (or no robot given)."""
    if robot is None:
        return True
    return robot.ws.is_inside_workspace(pt, robot.tri)


def _add_group(waypoints, candidates, home, hold_s, hold_home_s, robot):
    """
    Append all valid candidates, then a home hold.
    If no candidate is valid, the group (including home return) is skipped.
    """
    valid = [(pt, hold_s) for pt in candidates if _is_valid(pt, robot)]
    if valid:
        waypoints.extend(valid)
        waypoints.append((home.copy(), hold_home_s))
    else:
        print(f"[step_response] Skipped group of {len(candidates)} points (all outside workspace).")


def get_waypoints(robot=None):
    """Step inputs along ±X, ±Y, diagonal XY, and Z at multiple heights."""
    if robot is not None:
        z_home = robot.flowbot.l0 + robot.flowbot.lu
    else:
        z_home = 95.0

    z_min = z_home + Z_MIN_OFFSET
    z_max = z_home + Z_MAX_OFFSET
    z_vals  = np.linspace(z_min, z_max, N_POINTS_Z)
    xy_amps = np.linspace(XY_MAX_MM / N_POINTS_SAMPLED, XY_MAX_MM, N_POINTS_SAMPLED)
    z_amps  = np.linspace(z_min, z_max, N_POINTS_SAMPLED)

    home = np.array([0.0, 0.0, z_home])
    waypoints = [(home.copy(), HOLD_HOME_S)]

    directions = {
        "+X":   lambda amp, z: np.array([ amp,  0.0, z]),
        "-X":   lambda amp, z: np.array([-amp,  0.0, z]),
        "+Y":   lambda amp, z: np.array([ 0.0,  amp, z]),
        "-Y":   lambda amp, z: np.array([ 0.0, -amp, z]),
        "diag": lambda amp, z: np.array([ amp / np.sqrt(2), amp / np.sqrt(2), z]),
    }

    for _, fn in directions.items():
        for z in z_vals:
            candidates = [fn(amp, z) for amp in xy_amps]
            _add_group(waypoints, candidates, home, HOLD_S, HOLD_HOME_S, robot)

    # Z-only steps
    z_candidates = [np.array([0.0, 0.0, z]) for z in z_amps]
    _add_group(waypoints, z_candidates, home, HOLD_S, HOLD_HOME_S, robot)

    n_steps = sum(1 for pt, _ in waypoints
                  if not (pt[0] == 0.0 and pt[1] == 0.0 and pt[2] == z_home))
    print(f"[step_response] {n_steps} step waypoints generated (home visits excluded).")

    return waypoints
