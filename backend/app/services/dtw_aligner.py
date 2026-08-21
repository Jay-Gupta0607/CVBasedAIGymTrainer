"""Dynamic Time Warping for video synchronization."""

import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from app.services.pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)


class DTWAligner:
    """Align two video sequences using DTW on pose embeddings."""

    def __init__(self, pose_estimator: Optional[PoseEstimator] = None):
        self.pose_estimator = pose_estimator or PoseEstimator()

    def extract_pose_embeddings(
        self,
        frames: List[np.ndarray],
    ) -> np.ndarray:
        """Extract pose embeddings from video frames.

        Args:
            frames: List of BGR frames

        Returns:
            Embeddings array of shape (num_frames, 99) - 33 landmarks * 3 coords
        """
        embeddings = []
        estimator = self.pose_estimator

        for i, frame in enumerate(frames):
            try:
                landmarks_3d, visibility, _ = estimator.estimate(frame)

                # Only use landmarks with good visibility
                mask = visibility > 0.5
                if np.sum(mask) < 10:  # Not enough visible landmarks
                    # Use zero embedding
                    embeddings.append(np.zeros(99, dtype=np.float32))
                    continue

                # Flatten visible landmarks, pad invisible with zeros
                embedding = np.zeros(99, dtype=np.float32)
                for idx in range(33):
                    if visibility[idx] > 0.5:
                        embedding[idx*3:(idx+1)*3] = landmarks_3d[idx]

                embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to extract pose for frame {i}: {e}")
                embeddings.append(np.zeros(99, dtype=np.float32))

        return np.array(embeddings)

    def align_sequences(
        self,
        user_embeddings: np.ndarray,
        trainer_embeddings: np.ndarray,
        radius: int = 10,
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Align two pose embedding sequences using FastDTW.

        Args:
            user_embeddings: (N, 99) user pose embeddings
            trainer_embeddings: (M, 99) trainer pose embeddings
            radius: Sakoe-Chiba band radius for DTW constraint

        Returns:
            Tuple of (warping_path, normalized_distance)
            warping_path: List of (user_idx, trainer_idx) pairs
            normalized_distance: Average distance per step
        """
        # Ensure 2D arrays
        if user_embeddings.ndim == 1:
            user_embeddings = user_embeddings.reshape(1, -1)
        if trainer_embeddings.ndim == 1:
            trainer_embeddings = trainer_embeddings.reshape(1, -1)

        # Compute DTW
        distance, path = fastdtw(
            user_embeddings,
            trainer_embeddings,
            dist=euclidean,
            radius=radius,
        )

        # Path is list of (user_idx, trainer_idx)
        warping_path = [(int(i), int(j)) for i, j in path]

        # Normalize distance by path length
        normalized_distance = distance / len(path) if path else float('inf')

        logger.info(f"DTW alignment: {len(user_embeddings)} user frames -> "
                   f"{len(trainer_embeddings)} trainer frames, "
                   f"path length: {len(warping_path)}, "
                   f"normalized distance: {normalized_distance:.4f}")

        return warping_path, normalized_distance

    def get_frame_correspondences(
        self,
        warping_path: List[Tuple[int, int]],
        user_frame_indices: List[int],
        trainer_frame_indices: List[int],
    ) -> List[dict]:
        """Convert warping path to frame correspondences.

        Args:
            warping_path: List of (user_emb_idx, trainer_emb_idx)
            user_frame_indices: Original frame indices for user embeddings
            trainer_frame_indices: Original frame indices for trainer embeddings

        Returns:
            List of correspondence dicts with frame_ids and similarity
        """
        correspondences = []

        for user_emb_idx, trainer_emb_idx in warping_path:
            if (user_emb_idx < len(user_frame_indices) and
                trainer_emb_idx < len(trainer_frame_indices)):
                correspondences.append({
                    "user_frame_id": user_frame_indices[user_emb_idx],
                    "trainer_frame_id": trainer_frame_indices[trainer_emb_idx],
                    "user_embedding_idx": user_emb_idx,
                    "trainer_embedding_idx": trainer_emb_idx,
                })

        return correspondences

    def segment_repetitions(
        self,
        embeddings: np.ndarray,
        visibility_scores: np.ndarray,
        exercise_name: str,
    ) -> List[Tuple[int, int]]:
        """Segment pose sequence into repetitions.

        Args:
            embeddings: (N, 99) pose embeddings
            visibility_scores: (N, 33) visibility per frame
            exercise_name: Exercise type for specific logic

        Returns:
            List of (start_frame, end_frame) for each rep
        """
        # Use key joint angles to detect repetition cycles
        # This is a simplified version - in production, use exercise-specific logic

        # Calculate movement magnitude per frame
        diff = np.diff(embeddings, axis=0)
        movement = np.linalg.norm(diff, axis=1)

        # Smooth with moving average
        window = 5
        if len(movement) >= window:
            movement_smooth = np.convolve(movement, np.ones(window)/window, mode='valid')
        else:
            movement_smooth = movement

        # Find local minima (transition points between reps)
        from scipy.signal import argrelextrema
        minima = argrelextrema(movement_smooth, np.less)[0]

        # Filter minima by visibility (need good pose detection)
        valid_minima = []
        for m in minima:
            if m + window < len(visibility_scores):
                avg_vis = np.mean(visibility_scores[m:m+window] > 0.5)
                if avg_vis > 0.7:
                    valid_minima.append(m)

        # Create segments
        segments = []
        start = 0
        for end in valid_minima:
            if end - start > 10:  # Minimum rep length
                segments.append((start, end))
                start = end

        # Add final segment
        if len(embeddings) - start > 10:
            segments.append((start, len(embeddings) - 1))

        logger.info(f"Detected {len(segments)} repetitions for {exercise_name}")
        return segments


def compute_joint_angles(landmarks_3d: np.ndarray) -> dict:
    """Compute key joint angles from 3D landmarks.

    Args:
        landmarks_3d: (33, 3) array of 3D landmarks

    Returns:
        Dictionary of joint angles in degrees
    """
    def angle_between(p1, p2, p3):
        """Angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    angles = {}

    # Key joint angles for exercises
    # Left arm
    angles['left_elbow'] = angle_between(
        landmarks_3d[11], landmarks_3d[13], landmarks_3d[15]
    )
    angles['left_shoulder'] = angle_between(
        landmarks_3d[13], landmarks_3d[11], landmarks_3d[23]
    )

    # Right arm
    angles['right_elbow'] = angle_between(
        landmarks_3d[12], landmarks_3d[14], landmarks_3d[16]
    )
    angles['right_shoulder'] = angle_between(
        landmarks_3d[14], landmarks_3d[12], landmarks_3d[24]
    )

    # Left leg
    angles['left_knee'] = angle_between(
        landmarks_3d[23], landmarks_3d[25], landmarks_3d[27]
    )
    angles['left_hip'] = angle_between(
        landmarks_3d[25], landmarks_3d[23], landmarks_3d[11]
    )

    # Right leg
    angles['right_knee'] = angle_between(
        landmarks_3d[24], landmarks_3d[26], landmarks_3d[28]
    )
    angles['right_hip'] = angle_between(
        landmarks_3d[26], landmarks_3d[24], landmarks_3d[12]
    )

    # Torso
    angles['torso_lean'] = angle_between(
        landmarks_3d[11], landmarks_3d[23], landmarks_3d[24]
    )

    return angles


def compute_angle_differences(
    user_angles: dict,
    trainer_angles: dict,
) -> dict:
    """Compute differences between user and trainer joint angles.

    Args:
        user_angles: User joint angles
        trainer_angles: Trainer joint angles

    Returns:
        Dictionary of angle differences
    """
    return {
        joint: abs(user_angles.get(joint, 0) - trainer_angles.get(joint, 0))
        for joint in set(user_angles) | set(trainer_angles)
    }