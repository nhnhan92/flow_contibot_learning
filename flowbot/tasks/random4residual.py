"""
Task: Local-walk random waypoints for residual-model data collection.

Each successive waypoint is sampled within STEP_RADIUS_MM of the previous
point (instead of globally random).  This creates spatially-correlated
trajectories that better represent the continuous motions the robot performs,
which is more useful for training a sequence-based error model.

Sampling strategy
-----------------
  1. Start from a random initial point inside the workspace.
  2. For each subsequent point, draw a candidate from a sphere of radius
     STEP_RADIUS_MM centred on the current point, accept if inside workspace
     and Z bounds.  Retry up to MAX_STEP_TRIES times; if that fails, teleport
     to a fresh random point (so the walk doesn't get stuck in a corner).
"""
import numpy as np

TASK_NAME      = "random4residual"

SEED           = 17   # set to None for truly random each run
N_POINTS       = 20       # number of waypoints (excluding home at start/end)
Z_MIN_OFFSET   = 5.0      # min Z above home (mm)
Z_MAX_OFFSET   = 22.0     # max Z above home (mm)
HOLD_S         = 1.2        # hold time at each waypoint (s)
STEP_RADIUS_MM = 10.0     # max distance between consecutive waypoints (mm)
P_EXPLORE      = 0.15    # probability of global jump 
MAX_STEP_TRIES = 500      # retries before teleporting to a fresh point
MIN_DIST_MM     = 5.0      # minimum distance for a valid step (mm)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _random_in_workspace(robot, rng, mn, mx, max_tries=5000):
    """Sample one point uniformly inside the workspace hull within Z bounds."""
    for _ in range(max_tries):
        q = mn + (mx - mn) * rng.random(3)
        if robot.ws.is_inside_workspace(q, robot.tri):
            return q
    raise RuntimeError(
        f"Could not find a valid point inside workspace after {max_tries} tries."
    )


def _step_from(current, step_r, z_min, z_max, robot, rng, mn, mx):
    """
    Sample a point within a sphere of radius `step_r` around `current`,
    clipped to Z bounds and workspace hull.
    Returns the new point, or None if MAX_STEP_TRIES exceeded.
    """
    for _ in range(MAX_STEP_TRIES):
        direction = rng.standard_normal(3)
        norm = np.linalg.norm(direction)
        if norm < MIN_DIST_MM:
            continue
        direction /= norm
        r = step_r * rng.random() ** (1 / 3)   # cube-root for uniform volume
        candidate = current + direction * r

        # Clip Z
        candidate[2] = np.clip(candidate[2], z_min, z_max)

        # Bounding-box pre-filter
        if np.any(candidate < mn) or np.any(candidate > mx):
            continue

        if robot.ws.is_inside_workspace(candidate, robot.tri):
            return candidate

    return None   # failed → caller will teleport


def _sample_local_walk(robot, z_min, z_max, n_points, rng, mn, mx):
    """Build a list of `n_points` by local-walk sampling."""
    points = []

    current = _random_in_workspace(robot, rng, mn, mx)
    points.append(current.copy())

    while len(points) < n_points:
        # if rng.random() < P_EXPLORE:
        #     # Global jump: explore a completely different region
        #     nxt = _random_in_workspace(robot, rng, mn, mx)
        # else:
        nxt = _step_from(current, STEP_RADIUS_MM, z_min, z_max, robot, rng, mn, mx)
        if nxt is None:
            # Stuck in a corner — teleport to a fresh point
            nxt = _random_in_workspace(robot, rng, mn, mx)
        points.append(nxt.copy())
        current = nxt

    return points



# ─── public API ──────────────────────────────────────────────────────────────

def get_waypoints(robot=None, seed=SEED):
    rng = np.random.default_rng(seed)

    z_home = robot.flowbot.l0 + robot.flowbot.lu
    z_min  = z_home + Z_MIN_OFFSET
    z_max  = z_home + Z_MAX_OFFSET

    mn, mx = robot.bbox
    mn = mn.copy(); mx = mx.copy()
    mn[2] = max(mn[2], z_min)
    mx[2] = min(mx[2], z_max)

    if mn[2] >= mx[2]:
        raise ValueError(
            f"Z bounds [{z_min:.1f}, {z_max:.1f}] mm don't overlap with "
            f"workspace Z range [{robot.bbox[0][2]:.1f}, {robot.bbox[1][2]:.1f}] mm."
        )

    pts = _sample_local_walk(robot, z_min, z_max, N_POINTS, rng, mn, mx)

    home = np.array([0.0, 0.0,
                     z_home if robot is None else robot.flowbot.l0 + robot.flowbot.lu])
    waypoints = [(home.copy(), 1.0)]
    for pt in pts:
        waypoints.append((np.asarray(pt, dtype=float), HOLD_S))
    waypoints.append((home.copy(), 1.0))

    return waypoints
