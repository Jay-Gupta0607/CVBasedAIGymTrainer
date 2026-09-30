"""Dynamic Time Warping for video synchronization."""

import logging
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from app.services.pose_estimator import PoseEstimator
from app.services.pose_features import (  # noqa: F401  (re-exported: existing import path)
    compute_angle_differences,
    compute_joint_angles,
    pose_embedding,
)

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
            Embeddings array of shape (num_frames, 99) - 33 landmarks * 3 coords,
            hip-centred and in torso lengths (see pose_features.pose_embedding)
        """
        embeddings = []
        estimator = self.pose_estimator

        for i, frame in enumerate(frames):
            try:
                landmarks_3d, visibility, _ = estimator.estimate(frame)

                embedding = pose_embedding(landmarks_3d, visibility)

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

    def segment_repetitions_by_signal(
        self,
        signal: np.ndarray,
        extreme: str = "min",
        min_prominence: float = 25.0,
        min_distance: int = 8,
        smooth_window: int = 5,
    ) -> List[Tuple[int, int]]:
        """Split a sequence into repetitions using a joint-angle time series.

        One repetition = one cycle of the signal, counted at its working extreme (e.g.
        the knee-angle minimum at the bottom of a squat). Unlike movement-speed minima,
        which occur at both turning points of a rep, this counts each rep once.

        Args:
            signal: per-frame angle in degrees; NaN where not measured
            extreme: "min" if the rep is counted at the angle's lowest point, else "max"
            min_prominence: how far (degrees) the angle must swing to count as a rep
            min_distance: minimum frames between two reps
            smooth_window: moving-average window (frames) applied before peak finding

        Returns:
            One (start, end) frame window per repetition, in order. Windows meet at the
            midpoints between consecutive extremes and cover the whole sequence.
        """
        from scipy.signal import find_peaks

        x = np.asarray(signal, dtype=float)
        valid = np.isfinite(x)
        if valid.sum() < max(10, smooth_window):
            return []

        # Fill gaps by interpolation so a few unmeasured frames don't split a rep
        idx = np.arange(len(x))
        x = np.interp(idx, idx[valid], x[valid])

        if smooth_window > 1:
            pad = smooth_window // 2
            padded = np.pad(x, pad, mode="edge")
            x = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")

        oriented = -x if extreme == "min" else x
        peaks, _ = find_peaks(oriented, prominence=min_prominence, distance=min_distance)
        if len(peaks) == 0:
            return []

        bounds = [0] + [int((peaks[i] + peaks[i + 1]) // 2) for i in range(len(peaks) - 1)] + [len(x) - 1]
        segments = [(bounds[i], bounds[i + 1]) for i in range(len(peaks))]
        logger.info(f"Detected {len(segments)} repetitions from joint-angle signal ({extreme})")
        return segments

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
