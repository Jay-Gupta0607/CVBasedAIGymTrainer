"""Pose geometry shared by alignment and scoring: joint angles and normalisation.

Landmarks are BlazePose's 33 points as an (33, 3) array: x, y (and z) in the same
pixel scale, y pointing down. Visibility is an optional (33,) array of probabilities.
"""

from typing import Dict, Optional

import numpy as np

MIN_VISIBILITY = 0.5

# Landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

TORSO_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

# angle name -> (point A, vertex, point B): the angle at the vertex between A and B
JOINT_ANGLE_TRIPLETS = {
    "left_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "left_shoulder": (L_ELBOW, L_SHOULDER, L_HIP),
    "right_shoulder": (R_ELBOW, R_SHOULDER, R_HIP),
    "left_knee": (L_HIP, L_KNEE, L_ANKLE),
    "right_knee": (R_HIP, R_KNEE, R_ANKLE),
    "left_hip": (L_KNEE, L_HIP, L_SHOULDER),
    "right_hip": (R_KNEE, R_HIP, R_SHOULDER),
}


def angle_between(p1: np.ndarray, vertex: np.ndarray, p3: np.ndarray) -> float:
    """Angle in degrees at `vertex` formed by p1-vertex-p3 (0 = folded, 180 = straight)."""
    v1 = p1 - vertex
    v2 = p3 - vertex
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _visible(visibility: Optional[np.ndarray], indices, min_visibility: float) -> bool:
    if visibility is None:
        return True
    return bool(np.all(np.asarray(visibility)[list(indices)] >= min_visibility))


def compute_joint_angles(
    landmarks_3d: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    min_visibility: float = MIN_VISIBILITY,
) -> Dict[str, float]:
    """Compute key joint angles in degrees.

    Angles whose landmarks are not all visible are left out (when `visibility` is
    given), so callers can tell "not measurable" from "measured".

    `torso_lean` is the angle between the hip->shoulder line and the vertical in the
    image plane: 0 = upright, 90 = horizontal. (Depth lean is not observable from one
    camera.)
    """
    angles: Dict[str, float] = {}

    for name, (a, vertex, b) in JOINT_ANGLE_TRIPLETS.items():
        if _visible(visibility, (a, vertex, b), min_visibility):
            angles[name] = angle_between(landmarks_3d[a], landmarks_3d[vertex], landmarks_3d[b])

    if _visible(visibility, TORSO_POINTS, min_visibility):
        mid_shoulder = landmarks_3d[[L_SHOULDER, R_SHOULDER], :2].mean(axis=0)
        mid_hip = landmarks_3d[[L_HIP, R_HIP], :2].mean(axis=0)
        torso = mid_shoulder - mid_hip
        length = np.linalg.norm(torso)
        if length > 1e-6:
            up = np.array([0.0, -1.0])  # image y grows downward
            cos_lean = float(np.dot(torso, up) / length)
            angles["torso_lean"] = float(np.degrees(np.arccos(np.clip(cos_lean, -1.0, 1.0))))

    return angles


def compute_angle_differences(user_angles: Dict[str, float], trainer_angles: Dict[str, float]) -> Dict[str, float]:
    """Absolute per-joint differences, for joints measured in both poses."""
    return {
        joint: abs(user_angles[joint] - trainer_angles[joint])
        for joint in set(user_angles) & set(trainer_angles)
    }


def torso_length(landmarks_3d: np.ndarray) -> float:
    """Distance between the shoulder midpoint and the hip midpoint."""
    mid_shoulder = landmarks_3d[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
    mid_hip = landmarks_3d[[L_HIP, R_HIP]].mean(axis=0)
    return float(np.linalg.norm(mid_shoulder - mid_hip))


def normalize_pose(
    landmarks_3d: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    min_visibility: float = MIN_VISIBILITY,
) -> Optional[np.ndarray]:
    """Express a pose relative to the person: hip-centred, in units of torso length.

    This makes poses comparable across videos with different framing, distance and
    resolution. Returns None when the torso is not reliably visible.
    """
    if not _visible(visibility, TORSO_POINTS, min_visibility):
        return None
    scale = torso_length(landmarks_3d)
    if scale < 1e-6:
        return None
    mid_hip = landmarks_3d[[L_HIP, R_HIP]].mean(axis=0)
    return ((landmarks_3d - mid_hip) / scale).astype(np.float32)


def pose_embedding(
    landmarks_3d: np.ndarray,
    visibility: np.ndarray,
    min_visibility: float = MIN_VISIBILITY,
) -> np.ndarray:
    """99-d embedding for sequence alignment: normalised coordinates of visible
    landmarks, zeros elsewhere (all zeros when the torso is not visible)."""
    embedding = np.zeros(33 * 3, dtype=np.float32)
    normalized = normalize_pose(landmarks_3d, visibility, min_visibility)
    if normalized is None:
        return embedding
    for i in range(33):
        if visibility[i] >= min_visibility:
            embedding[i * 3 : (i + 1) * 3] = normalized[i]
    return embedding
