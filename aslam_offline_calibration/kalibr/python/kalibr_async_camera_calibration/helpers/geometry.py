'''

rotation_matrix_to_rotvec:
    Convert a 3x3 rotation matrix to a rotation vector (axis * angle).
    Uses the Rodrigues formula inverse.
    Matches the convention used by bsplines.BSplinePose (sm.RotationVector).

'''

import numpy as np

def rotation_matrix_to_rotvec(R):
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2.0 * np.sin(angle))
    return axis * angle