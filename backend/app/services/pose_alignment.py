"""Pose Direction Detection and Alignment Module.

Handles automatic detection of facing direction (left/right) and mirrors
trainer pose to match user's facing direction for pose transfer.
"""

import logging
from typing import Tuple, Optional, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# DWPose body keypoint indices (18 keypoints)
# Based on COCO-WholeBody format
class DWPOSE_IDX:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    NECK = 17


# Left-right keypoint pairs for mirroring (body only)
DWPOSE_LR_PAIRS = [
    (DWPOSE_IDX.LEFT_EYE, DWPOSE_IDX.RIGHT_EYE),
    (DWPOSE_IDX.LEFT_EAR, DWPOSE_IDX.RIGHT_EAR),
    (DWPOSE_IDX.LEFT_SHOULDER, DWPOSE_IDX.RIGHT_SHOULDER),
    (DWPOSE_IDX.LEFT_ELBOW, DWPOSE_IDX.RIGHT_ELBOW),
    (DWPOSE_IDX.LEFT_WRIST, DWPOSE_IDX.RIGHT_WRIST),
    (DWPOSE_IDX.LEFT_HIP, DWPOSE_IDX.RIGHT_HIP),
    (DWPOSE_IDX.LEFT_KNEE, DWPOSE_IDX.RIGHT_KNEE),
    (DWPOSE_IDX.LEFT_ANKLE, DWPOSE_IDX.RIGHT_ANKLE),
]

# Skeleton connections for drawing (body only - 18 keypoints)
DWPOSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (5, 11), (6, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (17, 5), (17, 6), (17, 11), (17, 12),  # Neck connections
]

# Extended pairs including face and hands (for full 130 keypoints)
DWPOSE_FULL_LR_PAIRS = [
    # Face (68 points, symmetric pairs) - approximate
    *[(18 + i, 18 + i + 34) for i in range(17)],  # Rough face symmetry
    # Body
    (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    # Hands (21 each)
    *[(86 + i, 107 + i) for i in range(21)],
]


class PoseDirectionAligner:
    """Detect facing direction and align trainer pose to user direction."""

    def __init__(self, min_keypoint_score: float = 0.3):
        """Initialize direction aligner.

        Args:
            min_keypoint_score: Minimum confidence to consider keypoint valid
        """
        self.min_score = min_keypoint_score

    def detect_facing_direction(
        self,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        method: Literal["torso", "shoulder_hip", "heuristic"] = "shoulder_hip",
    ) -> Literal["left", "right"]:
        """Determine if person faces 'left' or 'right'.

        Args:
            keypoints: (N, 2) keypoints array (at least 18 body keypoints)
            scores: (N,) optional confidence scores
            method: Detection method to use

        Returns:
            "left" or "right" indicating facing direction
        """
        if scores is not None:
            valid_mask = scores > self.min_score
        else:
            valid_mask = np.ones(len(keypoints), dtype=bool)

        # Need at least shoulders and hips
        required_idxs = [
            DWPOSE_IDX.LEFT_SHOULDER, DWPOSE_IDX.RIGHT_SHOULDER,
            DWPOSE_IDX.LEFT_HIP, DWPOSE_IDX.RIGHT_HIP
        ]
        if not all(valid_mask[i] for i in required_idxs):
            logger.warning("Insufficient keypoints for direction detection, defaulting to 'right'")
            return "right"

        left_shoulder = keypoints[DWPOSE_IDX.LEFT_SHOULDER]
        right_shoulder = keypoints[DWPOSE_IDX.RIGHT_SHOULDER]
        left_hip = keypoints[DWPOSE_IDX.LEFT_HIP]
        right_hip = keypoints[DWPOSE_IDX.RIGHT_HIP]

        if method == "torso":
            # Use torso vector (neck to mid-hip) cross hip vector
            # If neck not available, use mid-shoulder
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            mid_hip = (left_hip + right_hip) / 2
            torso_vec = mid_shoulder - mid_hip
            hip_vec = right_hip - left_hip
            cross_z = torso_vec[0] * hip_vec[1] - torso_vec[1] * hip_vec[0]

        elif method == "shoulder_hip":
            # Use shoulder vector cross hip vector
            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip
            cross_z = shoulder_vec[0] * hip_vec[1] - shoulder_vec[1] * hip_vec[0]

        else:  # heuristic
            # Simple heuristic: which shoulder is more visible/higher
            # In front view, left shoulder higher = facing right (camera perspective)
            # This is less reliable, use cross product methods above
            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip
            cross_z = shoulder_vec[0] * hip_vec[1] - shoulder_vec[1] * hip_vec[0]

        # Cross product Z > 0 means counter-clockwise (facing right in image coords)
        # Cross product Z < 0 means clockwise (facing left in image coords)
        direction = "right" if cross_z > 0 else "left"

        logger.debug(f"Direction detection: cross_z={cross_z:.2f} -> {direction}")
        return direction

    def mirror_pose_horizontally(
        self,
        keypoints: np.ndarray,
        scores: Optional[np.ndarray] = None,
        image_width: Optional[int] = None,
        full_keypoints: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Mirror keypoints horizontally around image center.

        Args:
            keypoints: (N, 2) keypoints array
            scores: (N,) optional confidence scores
            image_width: Image width for mirroring (uses max x if not provided)
            full_keypoints: If True, use full 130 keypoint pairs; else body only

        Returns:
            Tuple of (mirrored_keypoints, mirrored_scores)
        """
        mirrored_kpts = keypoints.copy()
        mirrored_scores = scores.copy() if scores is not None else None

        # Determine mirror axis
        if image_width is not None:
            mirror_x = image_width / 2
        else:
            # Use keypoint bounds
            mirror_x = np.max(keypoints[:, 0]) / 2 + np.min(keypoints[:, 0]) / 2

        # Flip X coordinates
        mirrored_kpts[:, 0] = 2 * mirror_x - keypoints[:, 0]

        # Swap left/right keypoint pairs
        pairs = DWPOSE_FULL_LR_PAIRS if full_keypoints else DWPOSE_LR_PAIRS
        for left_idx, right_idx in pairs:
            if left_idx < len(mirrored_kpts) and right_idx < len(mirrored_kpts):
                # Swap positions
                mirrored_kpts[[left_idx, right_idx]] = mirrored_kpts[[right_idx, left_idx]]
                # Swap scores
                if mirrored_scores is not None:
                    mirrored_scores[[left_idx, right_idx]] = mirrored_scores[[right_idx, left_idx]]

        return mirrored_kpts, mirrored_scores

    def align_trainer_to_user(
        self,
        user_keypoints: np.ndarray,
        user_scores: np.ndarray,
        trainer_keypoints: np.ndarray,
        trainer_scores: np.ndarray,
        user_image_shape: Tuple[int, int],
        trainer_image_shape: Tuple[int, int],
        full_keypoints: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Align trainer pose to match user's facing direction.

        Args:
            user_keypoints: (N, 2) user keypoints
            user_scores: (N,) user keypoint scores
            trainer_keypoints: (N, 2) trainer keypoints
            trainer_scores: (N,) trainer keypoint scores
            user_image_shape: (H, W) of user image
            trainer_image_shape: (H, W) of trainer image
            full_keypoints: Whether to use full 130 keypoints

        Returns:
            Tuple of (aligned_trainer_keypoints, aligned_trainer_scores, alignment_info)
        """
        # Detect directions
        user_dir = self.detect_facing_direction(user_keypoints, user_scores)
        trainer_dir = self.detect_facing_direction(trainer_keypoints, trainer_scores)

        alignment_info = {
            "user_direction": user_dir,
            "trainer_direction": trainer_dir,
            "mirrored": False,
        }

        logger.info(f"Direction alignment: user={user_dir}, trainer={trainer_dir}")

        if user_dir != trainer_dir:
            # Mirror trainer to match user
            aligned_kpts, aligned_scores = self.mirror_pose_horizontally(
                trainer_keypoints,
                trainer_scores,
                image_width=trainer_image_shape[1],
                full_keypoints=full_keypoints,
            )
            alignment_info["mirrored"] = True
            logger.info("Trainer pose mirrored to match user direction")
            return aligned_kpts, aligned_scores, alignment_info

        # No mirroring needed
        return trainer_keypoints, trainer_scores, alignment_info

    def compute_pose_similarity(
        self,
        keypoints1: np.ndarray,
        scores1: np.ndarray,
        keypoints2: np.ndarray,
        scores2: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """Compute similarity between two poses (for debugging/validation).

        Args:
            keypoints1, keypoints2: (N, 2) keypoint arrays
            scores1, scores2: (N,) confidence scores
            normalize: Whether to normalize by image size

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Use only body keypoints (18)
        body_indices = list(range(18))

        kpts1 = keypoints1[body_indices]
        kpts2 = keypoints2[body_indices]
        vis1 = scores1[body_indices] > self.min_score
        vis2 = scores2[body_indices] > self.min_score

        # Only compare jointly visible keypoints
        joint_vis = vis1 & vis2
        if np.sum(joint_vis) < 5:
            return 0.0

        kpts1_vis = kpts1[joint_vis]
        kpts2_vis = kpts2[joint_vis]

        if normalize:
            # Normalize by torso size
            torso1 = np.linalg.norm(
                kpts1[DWPOSE_IDX.LEFT_SHOULDER] - kpts1[DWPOSE_IDX.LEFT_HIP]
            )
            torso2 = np.linalg.norm(
                kpts2[DWPOSE_IDX.LEFT_SHOULDER] - kpts2[DWPOSE_IDX.LEFT_HIP]
            )
            scale = (torso1 + torso2) / 2
            if scale > 0:
                kpts1_vis = kpts1_vis / scale
                kpts2_vis = kpts2_vis / scale

        # Mean Euclidean distance
        distances = np.linalg.norm(kpts1_vis - kpts2_vis, axis=1)
        mean_dist = np.mean(distances)

        # Convert to similarity (0-1)
        similarity = np.exp(-mean_dist / 50.0)  # 50px characteristic scale
        return float(similarity)


def create_controlnet_conditioning(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: Tuple[int, int],
    line_thickness: int = 3,
    point_radius: int = 5,
    min_score: float = 0.3,
) -> np.ndarray:
    """Create ControlNet conditioning image for DWPOse.

    Black background with white skeleton overlay - standard format for ControlNet.

    Args:
        keypoints: (N, 2) keypoints (at least 18 body)
        scores: (N,) confidence scores
        image_shape: (H, W) output image size
        line_thickness: Skeleton line thickness
        point_radius: Keypoint circle radius
        min_score: Minimum score to draw

    Returns:
        Conditioning image (H, W, 3) - uint8
    """
    h, w = image_shape
    cond_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw body skeleton
    for start, end in DWPOSE_SKELETON:
        if start < len(scores) and end < len(scores):
            if scores[start] > min_score and scores[end] > min_score:
                pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
                pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
                cv2.line(cond_img, pt1, pt2, (255, 255, 255), line_thickness)

    # Draw keypoints
    for i in range(min(18, len(keypoints))):
        if scores[i] > min_score:
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            cv2.circle(cond_img, (x, y), point_radius, (255, 255, 255), -1)

    return cond_img


# Global instance
_pose_aligner: Optional[PoseDirectionAligner] = None


def get_pose_aligner() -> PoseDirectionAligner:
    """Get or create global pose aligner instance."""
    global _pose_aligner
    if _pose_aligner is None:
        _pose_aligner = PoseDirectionAligner()
    return _pose_aligner