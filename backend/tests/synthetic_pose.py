"""Synthetic side-view skeletons with known joint angles, for testing without footage.

`squat_skeleton` builds a (33, 3) landmark array of a person seen from the side; both
legs/arms coincide. Coordinates are image-style (y grows downward). Only the landmarks
used for angles and normalisation are placed; the rest stay at the hip.
"""

from typing import Iterable, List, Tuple

import numpy as np

from app.services.pose_features import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)

SHIN, THIGH, TORSO, UPPER_ARM, FOREARM = 0.45, 0.45, 0.55, 0.30, 0.28  # metres-ish


def _rotate(v: np.ndarray, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])


def squat_skeleton(
    knee_angle: float,
    torso_lean: float = 15.0,
    scale: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
    elbow_angle: float = 180.0,
) -> np.ndarray:
    """Side-view pose whose knee joint angle is `knee_angle` degrees (180 = standing).

    The person faces +x. The shin leans forward as the knee bends; the hip goes back.
    """
    ankle = np.array([0.0, 0.0])
    shin_lean = (180.0 - knee_angle) * 0.45
    knee = ankle + SHIN * np.array([np.sin(np.radians(shin_lean)), -np.cos(np.radians(shin_lean))])

    down_from_knee = (ankle - knee) / np.linalg.norm(ankle - knee)
    candidates = [knee + THIGH * _rotate(down_from_knee, s * knee_angle) for s in (+1, -1)]
    hip = min(candidates, key=lambda h: h[0])  # hip goes behind the knee

    lean = np.radians(torso_lean)
    shoulder = hip + TORSO * np.array([np.sin(lean), -np.cos(lean)])

    # Arms hang down from the shoulder; elbow_angle folds the forearm forward
    elbow = shoulder + UPPER_ARM * np.array([0.0, 1.0])
    forearm_dir = _rotate(np.array([0.0, 1.0]), -(180.0 - elbow_angle))
    wrist = elbow + FOREARM * forearm_dir

    landmarks = np.tile(np.append(hip, 0.0), (33, 1)).astype(np.float64)
    for idx, point in {
        L_ANKLE: ankle, R_ANKLE: ankle, L_KNEE: knee, R_KNEE: knee, L_HIP: hip, R_HIP: hip,
        L_SHOULDER: shoulder, R_SHOULDER: shoulder, L_ELBOW: elbow, R_ELBOW: elbow,
        L_WRIST: wrist, R_WRIST: wrist,
    }.items():
        landmarks[idx, :2] = point

    landmarks[:, :2] = landmarks[:, :2] * scale + np.array(offset)
    landmarks[:, 2] = 0.0
    return landmarks.astype(np.float32)


def knee_cycle(reps: int, frames_per_rep: int, bottom: float, top: float = 175.0) -> List[float]:
    """Knee angle over `reps` squats: starts standing, dips to `bottom`, returns."""
    t = np.arange(reps * frames_per_rep)
    mid, amp = (top + bottom) / 2.0, (top - bottom) / 2.0
    return list(mid + amp * np.cos(2 * np.pi * t / frames_per_rep))


def sequence(knee_angles: Iterable[float], **kwargs) -> List[np.ndarray]:
    return [squat_skeleton(k, **kwargs) for k in knee_angles]


VISIBLE = np.full(33, 0.99, dtype=np.float32)
