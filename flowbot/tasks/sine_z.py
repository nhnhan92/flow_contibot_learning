"""
Task: Sine wave along Z (up-down bending).
Useful for measuring vertical compliance and tracking accuracy.
"""
import numpy as np

TASK_NAME = "sine_z"

def get_waypoints(robot=None):
    """
    Generates a sine-wave trajectory along Z at zero XY.
    The robot bends/extends repeatedly.
    """
    z_home = 105.0
    
    n_points   = 50         # samples per cycle
    amplitude  = 20.0        # mm peak amplitude
    hold_s     = 0.4        # hold per waypoint (s)

    waypoints = []
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    for a in angles:
        z = z_home + amplitude * np.sin(a)
        pc = np.array([0.0, 0.0, z], dtype=float)
        waypoints.append((pc, hold_s))
    print(waypoints)
    return waypoints
