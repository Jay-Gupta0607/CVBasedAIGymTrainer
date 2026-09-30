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


MODEL_INPUT_SIZE = 256  # the landmark model takes a 256x256 crop
NUM_BODY_LANDMARKS = 33  # the model emits 39 (33 body + 6 auxiliary); only body is used

# A square region of the frame fed to the model: (center_x, center_y, side) in pixels.
# It may extend past the frame edges (that area is zero-padded).
Roi = Tuple[float, float, float]


def letterbox_roi(height: int, width: int) -> Roi:
    """Default ROI: the whole frame, padded to a square."""
    return width / 2.0, height / 2.0, float(max(height, width))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


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

    def preprocess(self, frame: np.ndarray, roi: Optional[Roi] = None) -> np.ndarray:
        """Crop `roi` (default: whole frame, letterboxed) to 256x256 for the model.

        Args:
            frame: BGR image (H, W, 3) from OpenCV
            roi: Optional (center_x, center_y, side) square region in pixels

        Returns:
            Tensor (1, 256, 256, 3): RGB, NHWC, normalised to [0, 1]
        """
        h, w = frame.shape[:2]
        cx, cy, side = roi if roi is not None else letterbox_roi(h, w)

        # Affine map: original pixels -> 256x256 crop. Area outside the frame is black,
        # which reproduces the zero-padding of a letterbox.
        scale = MODEL_INPUT_SIZE / side
        matrix = np.array(
            [
                [scale, 0.0, -(cx - side / 2.0) * scale],
                [0.0, scale, -(cy - side / 2.0) * scale],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(
            frame,
            matrix,
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int],
        roi: Optional[Roi] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map raw model outputs back to original-image coordinates.

        The model's first output is (1, 195) = 39 landmarks x (x, y, z, visibility,
        presence). x/y/z are in 256x256 crop pixels; visibility/presence are logits.

        Args:
            outputs: Model outputs
            original_shape: (height, width) of the frame that was preprocessed
            roi: The ROI that was passed to `preprocess` (default: letterbox)

        Returns:
            landmarks_3d: (33, 3) - x, y in original pixels; z in the same pixel scale
                (so 3D angles do not depend on video resolution). Not clipped to the
                frame: landmarks the model places off-screen keep their position.
            visibility: (33,) - probability in [0, 1]
        """
        h, w = original_shape
        cx, cy, side = roi if roi is not None else letterbox_roi(h, w)

        landmarks = outputs[0][0].reshape(-1, 5)[:NUM_BODY_LANDMARKS]

        # Inverse of the crop: 256-space -> ROI pixels -> original image pixels
        to_pixels = side / MODEL_INPUT_SIZE
        x = cx - side / 2.0 + landmarks[:, 0] * to_pixels
        y = cy - side / 2.0 + landmarks[:, 1] * to_pixels
        z = landmarks[:, 2] * to_pixels
        visibility = _sigmoid(landmarks[:, 3])

        landmarks_3d = np.stack([x, y, z], axis=1).astype(np.float32)
        return landmarks_3d, visibility.astype(np.float32)

    def estimate(
        self, frame: np.ndarray, roi: Optional[Roi] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Estimate pose from a single frame.

        Args:
            frame: BGR image (H, W, 3)
            roi: Optional square region to run the model on (default: whole frame).
                The model was trained on person-centred crops, so a tight full-body ROI
                is more accurate than the whole frame when the person is small.

        Returns:
            Tuple of (landmarks_3d, visibility, metadata)
            landmarks_3d: (33, 3) array
            visibility: (33,) array
            metadata: dict with preprocessing info
        """
        if not self._initialized:
            self.initialize()

        original_shape = frame.shape[:2]
        input_tensor = self.preprocess(frame, roi)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        landmarks_3d, visibility = self.postprocess(outputs, original_shape, roi)

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