"""
Task: Random waypoints within the robot's 3-D workspace volume.

When a robot object is available, points are rejection-sampled inside the
convex-hull workspace (built from the forward kinematics) while also
satisfying the Z bounds below.  Without a robot object the task falls back
to uniform sampling inside a cylinder.
"""
import numpy as np

TASK_NAME    = "random"

SEED         = 8      # set to None for truly random each run
N_POINTS     = 40
Z_MIN_OFFSET = 5.0      # min Z above home (mm)
Z_MAX_OFFSET = 23.0     # max Z above home (mm)
HOLD_S       = 1      # hold time at each waypoint (s)

# Fallback cylinder (used only when robot=None)
RADIUS_MM    = 17.5     # max radial offset in XY (mm)


def _sample_in_workspace(robot, z_min, z_max, n_points, rng, max_tries=10000):
    """
    Rejection-sample `n_points` inside the robot workspace hull,
    restricted to z ∈ [z_min, z_max].
    """
    points = []
    mn, mx = robot.bbox
    # Tighten the Z range of the bounding box
    mn = mn.copy(); mx = mx.copy()
    mn[2] = max(mn[2], z_min)
    mx[2] = min(mx[2], z_max)

    if mn[2] >= mx[2]:
        raise ValueError(
            f"Z bounds [{z_min:.1f}, {z_max:.1f}] mm don't overlap with "
            f"workspace Z range [{robot.bbox[0][2]:.1f}, {robot.bbox[1][2]:.1f}] mm."
        )

    attempts = 0
    while len(points) < n_points:
        if attempts >= max_tries:
            raise RuntimeError(
                f"Could only sample {len(points)}/{n_points} points inside "
                f"workspace after {max_tries} tries. Relax Z bounds or reduce N_POINTS."
            )
        q = mn + (mx - mn) * rng.random(3)
        if robot.ws.is_inside_workspace(q, robot.tri):
            points.append(q)
        attempts += 1

    return points


def get_waypoints(robot=None, seed=SEED):
    rng = np.random.default_rng(seed)

    if robot is not None:
        z_home = robot.flowbot.l0 + robot.flowbot.lu
        z_min  = z_home + Z_MIN_OFFSET
        z_max  = z_home + Z_MAX_OFFSET

        pts = _sample_in_workspace(robot, z_min, z_max, N_POINTS, rng)
    else:
        # Fallback: cylinder sampling (no hull available)
        z_home = 95.0
        z_min  = z_home + Z_MIN_OFFSET
        z_max  = z_home + Z_MAX_OFFSET

        angles = rng.uniform(0, 2 * np.pi, N_POINTS)
        radii  = RADIUS_MM * np.sqrt(rng.uniform(0, 1, N_POINTS))
        z_vals = rng.uniform(z_min, z_max, N_POINTS)
        pts = [np.array([r * np.cos(a), r * np.sin(a), z], dtype=float)
               for r, a, z in zip(radii, angles, z_vals)]

    home = np.array([0.0, 0.0, z_home if robot is None
                     else robot.flowbot.l0 + robot.flowbot.lu])
    waypoints = [(home.copy(), 1.0)]
    for pt in pts:
        waypoints.append((np.asarray(pt, dtype=float), HOLD_S))
    waypoints.append((home.copy(), 1.0))

    return waypoints
