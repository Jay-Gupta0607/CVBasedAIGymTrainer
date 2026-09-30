"""MediaPipe Pose Estimation using ONNX Runtime."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings

logger = logging.getLogger(__name__)


# MediaPipe Pose landmark indices (33 landmarks)
POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Keypoint pairs for drawing skeleton
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),  # Mouth
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
]


class PoseEstimator:
    """MediaPipe BlazePose estimation using ONNX Runtime."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.POSE_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize ONNX Runtime session."""
        if self._initialized:
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found at {model_path}")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Create session with providers
        providers = settings.onnx_providers_list
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]
            logger.warning("No GPU providers available, falling back to CPU")

        logger.info(f"Loading pose model with providers: {providers}")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Pose model loaded. Input: {self.input_name}, Outputs: {self.output_names}")
        self._initialized = True

    def is_healthy(self) -> bool:
        """Check if pose estimator is initialized and ready without reloading model."""
        return self._initialized and self.session is not None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for pose estimation.

        Args:
            frame: BGR image (H, W, 3) from OpenCV

        Returns:
            Preprocessed tensor (1, 256, 256, 3) for MediaPipe (NHWC, [0, 1])
        """
        # MediaPipe BlazePose expects RGB, 256x256, normalized to [0, 1], NHWC
        h, w = frame.shape[:2]
        size = max(h, w)

        # Pad to square
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = frame

        # Resize to 256x256
        resized = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] (the landmark model's TFLite input normalization)
        normalized = (rgb.astype(np.float32) / 255.0)

        # Add batch dimension (keep NHWC layout — the raw .tflite export wants it)
        return np.expand_dims(normalized, axis=0)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess model outputs.

        Args:
            outputs: Model outputs [landmarks_flat, presence, segmentation, ...]
                     landmarks_flat: (1, 195) = 39 landmarks × 5 (x,y,z,vis,pres)
            original_shape: (height, width) of original frame

        Returns:
            Tuple of (landmarks_3d, visibility)
            landmarks_3d: (33, 3) - x, y, z in original image coordinates (first 33 of 39)
            visibility: (33,) - visibility score for each landmark
        """
        # Raw output: (1, 195) = 39 × 5 values
        landmarks_flat = outputs[0][0]  # (195,)
        landmarks = landmarks_flat.reshape(39, 5)  # (39, 5): x, y, z, visibility, presence

        # Take only first 33 (body landmarks, skip 6 hand landmarks)
        landmarks = landmarks[:33]  # (33, 5)

        h, w = original_shape

        # Extract coordinates (already in absolute pixel coordinates from the model)
        x = landmarks[:, 0]  # absolute x
        y = landmarks[:, 1]  # absolute y
        z = landmarks[:, 2]  # depth
        visibility = landmarks[:, 3]  # confidence

        # Coordinates are in 256x256 space; scale back to original image
        # The model outputs in the padded 256x256 space
        size = max(h, w)
        x_offset = (size - w) / 2
        y_offset = (size - h) / 2

        # Remove padding offset and scale to original image
        x = (x - x_offset) * w / size
        y = (y - y_offset) * h / size

        # Clamp to image bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        landmarks_3d = np.stack([x, y, z], axis=1)

        return landmarks_3d, visibility

    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Estimate pose from a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Tuple of (landmarks_3d, visibility, metadata)
            landmarks_3d: (33, 3) array
            visibility: (33,) array
            metadata: dict with preprocessing info
        """
        if not self._initialized:
            self.initialize()

        original_shape = frame.shape[:2]
        input_tensor = self.preprocess(frame)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        landmarks_3d, visibility = self.postprocess(outputs, original_shape)

        metadata = {
            "input_shape": input_tensor.shape,
            "original_shape": original_shape,
        }

        return landmarks_3d, visibility, metadata

    def estimate_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Estimate pose for multiple frames.

        Args:
            frames: List of BGR images

        Returns:
            List of (landmarks_3d, visibility, metadata) tuples
        """
        results = []
        for frame in frames:
            try:
                result = self.estimate(frame)
                results.append(result)
            except Exception as e:
                logger.error(f"Pose estimation failed for frame: {e}")
                # Return empty landmarks
                h, w = frame.shape[:2]
                results.append((
                    np.zeros((33, 3), dtype=np.float32),
                    np.zeros(33, dtype=np.float32),
                    {"error": str(e), "original_shape": (h, w)}
                ))
        return results

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        visibility: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw pose landmarks on frame.

        Args:
            frame: BGR image to draw on
            landmarks: (33, 3) landmark coordinates
            visibility: (33,) visibility scores
            threshold: Minimum visibility to draw

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw connections
        for start, end in SKELETON_CONNECTIONS:
            if visibility[start] > threshold and visibility[end] > threshold:
                pt1 = (int(landmarks[start, 0]), int(landmarks[start, 1]))
                pt2 = (int(landmarks[end, 0]), int(landmarks[end, 1]))
                cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for i, (x, y, _) in enumerate(landmarks):
            if visibility[i] > threshold:
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1)

        return annotated


# Global instance for reuse
_pose_estimator: Optional[PoseEstimator] = None


def get_pose_estimator() -> PoseEstimator:
    """Get or create global pose estimator instance."""
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator