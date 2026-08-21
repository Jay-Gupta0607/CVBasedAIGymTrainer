"""DWPose Pose Estimation using ONNX Runtime.

DWPose provides 130+ keypoints (body 18 + face 68 + hands 42) vs MediaPipe's 33.
Better suited for ControlNet conditioning in pose-transfer generation.

ONNX model source: https://github.com/IDEA-Research/DWPose
Converted from: https://huggingface.co/yzd-v/DWPose
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings

logger = logging.getLogger(__name__)


# DWPose keypoint indices (COCO-WholeBody format)
# Body: 18 keypoints (0-17), Face: 68 keypoints (18-85), Hands: 42 keypoints (86-127)
DWPOSE_BODY_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "neck": 17,  # Additional body keypoint
}

# Face keypoints: 18-85 (68 points, MediaPipe face mesh compatible)
# Hand keypoints: 86-127 (21 per hand, MediaPipe hand compatible)

# Skeleton connections for drawing (body only - 18 keypoints)
DWPOSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (5, 11), (6, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (17, 5), (17, 6), (17, 11), (17, 12),  # Neck connections
]


class DWPoseEstimator:
    """DWPose estimation using ONNX Runtime.

    Outputs 130 keypoints:
    - 18 body keypoints (COCO format)
    - 68 face keypoints
    - 42 hand keypoints (21 per hand)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.DWPOSE_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self._initialized = False
        self.input_size = (288, 384)  # DWPose default input size (H, W)

    def initialize(self) -> None:
        """Initialize ONNX Runtime session."""
        if self._initialized:
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"DWPose model not found at {model_path}. "
                f"Run `python backend/scripts/download_dwpose.py` to download."
            )

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Create session with providers
        providers = settings.dwpose_onnx_providers_list
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]
            logger.warning("No GPU providers available for DWPose, falling back to CPU")

        logger.info(f"Loading DWPose model with providers: {providers}")

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"DWPose model loaded. Input: {self.input_name}, Outputs: {self.output_names}")
        self._initialized = True

    def is_healthy(self) -> bool:
        """Check if pose estimator is initialized and ready."""
        return self._initialized and self.session is not None

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Preprocess frame for DWPose estimation.

        Args:
            frame: BGR image (H, W, 3) from OpenCV

        Returns:
            Tuple of (preprocessed tensor (1, 3, H, W), metadata dict)
        """
        h, w = frame.shape[:2]

        # Letterbox resize to maintain aspect ratio
        target_h, target_w = self.input_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # Gray padding
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] then to [-1, 1] (ImageNet style)
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # HWC to CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        input_tensor = np.expand_dims(chw, axis=0)

        metadata = {
            "input_shape": input_tensor.shape,
            "original_shape": (h, w),
            "scale": scale,
            "padding": (y_offset, x_offset),
            "resized_shape": (new_h, new_w),
        }

        return input_tensor, metadata

    def postprocess(
        self,
        outputs: List[np.ndarray],
        metadata: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess model outputs.

        Args:
            outputs: Model outputs [keypoints, scores]
            metadata: Preprocessing metadata

        Returns:
            Tuple of (keypoints_2d, scores)
            keypoints_2d: (130, 2) - x, y in original image coordinates
            scores: (130,) - confidence score for each keypoint
        """
        # DWPose ONNX outputs: keypoints (1, 130, 2), scores (1, 130)
        keypoints = outputs[0][0]  # (130, 2) - normalized to input size
        scores = outputs[1][0] if len(outputs) > 1 else np.ones(130, dtype=np.float32)

        h, w = metadata["original_shape"]
        scale = metadata["scale"]
        y_offset, x_offset = metadata["padding"]

        # Denormalize from input size to original image coordinates
        # Keypoints are in input coordinate space (288, 384)
        keypoints[:, 0] = (keypoints[:, 0] - x_offset) / scale
        keypoints[:, 1] = (keypoints[:, 1] - y_offset) / scale

        # Clamp to image bounds
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h - 1)

        return keypoints, scores

    def estimate(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Estimate pose from a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Tuple of (keypoints_2d, scores, metadata)
            keypoints_2d: (130, 2) array
            scores: (130,) array
            metadata: dict with preprocessing info
        """
        if not self._initialized:
            self.initialize()

        input_tensor, metadata = self.preprocess(frame)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        keypoints, scores = self.postprocess(outputs, metadata)

        metadata["num_keypoints"] = len(keypoints)
        metadata["visible_keypoints"] = int(np.sum(scores > 0.3))

        return keypoints, scores, metadata

    def estimate_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Estimate pose for multiple frames."""
        results = []
        for frame in frames:
            try:
                result = self.estimate(frame)
                results.append(result)
            except Exception as e:
                logger.error(f"DWPose estimation failed for frame: {e}")
                h, w = frame.shape[:2]
                results.append((
                    np.zeros((130, 2), dtype=np.float32),
                    np.zeros(130, dtype=np.float32),
                    {"error": str(e), "original_shape": (h, w)}
                ))
        return results

    def get_body_keypoints(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Extract body keypoints (18) from full 130 keypoints."""
        body_indices = list(range(18))
        return keypoints[body_indices], scores[body_indices]

    def get_face_keypoints(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Extract face keypoints (68) from full 130 keypoints."""
        face_indices = list(range(18, 86))
        return keypoints[face_indices], scores[face_indices]

    def get_hand_keypoints(self, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract hand keypoints (21 each) from full 130 keypoints."""
        left_hand = keypoints[86:107], scores[86:107]
        right_hand = keypoints[107:128], scores[107:128]
        return left_hand, right_hand

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.3,
        draw_face: bool = False,
        draw_hands: bool = False,
    ) -> np.ndarray:
        """Draw pose skeleton on frame.

        Args:
            frame: BGR image to draw on
            keypoints: (130, 2) keypoints
            scores: (130,) confidence scores
            threshold: Minimum score to draw
            draw_face: Whether to draw face mesh
            draw_hands: Whether to draw hand skeletons

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw body skeleton
        for start, end in DWPOSE_SKELETON:
            if scores[start] > threshold and scores[end] > threshold:
                pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
                pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
                cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        # Draw body keypoints
        for i in range(18):
            if scores[i] > threshold:
                x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

        # Draw face keypoints (optional)
        if draw_face:
            for i in range(18, 86):
                if scores[i] > threshold:
                    x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                    cv2.circle(annotated, (x, y), 1, (255, 0, 0), -1)

        # Draw hand keypoints (optional)
        if draw_hands:
            for i in range(86, 128):
                if scores[i] > threshold:
                    x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                    cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)

        return annotated

    def create_controlnet_image(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        image_shape: Tuple[int, int],
        threshold: float = 0.3,
    ) -> np.ndarray:
        """Create ControlNet conditioning image (black background with white skeleton).

        This is the format expected by ControlNet-DWPOse for Flux.

        Args:
            keypoints: (130, 2) keypoints
            scores: (130,) confidence scores
            image_shape: (H, W) of target image
            threshold: Minimum score to draw

        Returns:
            ControlNet conditioning image (H, W, 3) - black bg, white lines
        """
        h, w = image_shape
        control_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw body skeleton in white
        for start, end in DWPOSE_SKELETON:
            if scores[start] > threshold and scores[end] > threshold:
                pt1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
                pt2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
                cv2.line(control_img, pt1, pt2, (255, 255, 255), 3)

        # Draw keypoints as white circles
        for i in range(18):
            if scores[i] > threshold:
                x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                cv2.circle(control_img, (x, y), 5, (255, 255, 255), -1)

        return control_img


# Global instance for reuse
_dwpose_estimator: Optional[DWPoseEstimator] = None


def get_dwpose_estimator() -> DWPoseEstimator:
    """Get or create global DWPose estimator instance."""
    global _dwpose_estimator
    if _dwpose_estimator is None:
        _dwpose_estimator = DWPoseEstimator()
    return _dwpose_estimator