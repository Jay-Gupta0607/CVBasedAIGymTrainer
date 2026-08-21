"""Form scoring using LightGBM ONNX model."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from app.config import settings

logger = logging.getLogger(__name__)


class FormScorer:
    """LightGBM-based form scoring using ONNX Runtime."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.SCORER_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self._initialized = False
        self._feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        """Get feature names in correct order (matches training)."""
        # 33 landmarks * 3 coords * 2 (user + trainer) = 198
        # Plus angle differences = ~20
        # Total ~220 features
        features = []

        # User landmarks (x, y, z for each of 33)
        for i in range(33):
            features.extend([f"user_lm_{i}_x", f"user_lm_{i}_y", f"user_lm_{i}_z"])

        # Trainer landmarks (x, y, z for each of 33)
        for i in range(33):
            features.extend([f"trainer_lm_{i}_x", f"trainer_lm_{i}_y", f"trainer_lm_{i}_z"])

        # Joint angle differences
        angle_names = [
            'left_elbow', 'left_shoulder', 'right_elbow', 'right_shoulder',
            'left_knee', 'left_hip', 'right_knee', 'right_hip', 'torso_lean'
        ]
        for angle in angle_names:
            features.append(f"angle_diff_{angle}")

        # Visibility scores
        for i in range(33):
            features.append(f"user_vis_{i}")
        for i in range(33):
            features.append(f"trainer_vis_{i}")

        return features

    def initialize(self) -> None:
        """Initialize ONNX Runtime session."""
        if self._initialized:
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(f"Scorer model not found at {model_path}, using dummy scorer")
            self._initialized = True
            return

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = settings.ONNX_PROVIDERS
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Form scorer loaded. Input: {self.input_name}, Output: {self.output_name}")
        self._initialized = True

    def prepare_features(
        self,
        user_landmarks: np.ndarray,  # (33, 3)
        user_visibility: np.ndarray,  # (33,)
        trainer_landmarks: np.ndarray,  # (33, 3)
        trainer_visibility: np.ndarray,  # (33,)
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
    ) -> np.ndarray:
        """Prepare feature vector for scoring.

        Args:
            user_landmarks: User pose landmarks (33, 3)
            user_visibility: User landmark visibility (33,)
            trainer_landmarks: Trainer pose landmarks (33, 3)
            trainer_visibility: Trainer landmark visibility (33,)
            user_angles: User joint angles
            trainer_angles: Trainer joint angles

        Returns:
            Feature vector (1, n_features)
        """
        features = []

        # User landmarks
        for i in range(33):
            features.extend(user_landmarks[i].tolist())

        # Trainer landmarks
        for i in range(33):
            features.extend(trainer_landmarks[i].tolist())

        # Angle differences
        angle_names = [
            'left_elbow', 'left_shoulder', 'right_elbow', 'right_shoulder',
            'left_knee', 'left_hip', 'right_knee', 'right_hip', 'torso_lean'
        ]
        for angle in angle_names:
            user_val = user_angles.get(angle, 0.0)
            trainer_val = trainer_angles.get(angle, 0.0)
            features.append(abs(user_val - trainer_val))

        # Visibility
        features.extend(user_visibility.tolist())
        features.extend(trainer_visibility.tolist())

        return np.array(features, dtype=np.float32).reshape(1, -1)

    def score_frame(
        self,
        user_landmarks: np.ndarray,
        user_visibility: np.ndarray,
        trainer_landmarks: np.ndarray,
        trainer_visibility: np.ndarray,
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
    ) -> Tuple[int, Dict[str, float]]:
        """Score a single frame pair.

        Returns:
            Tuple of (error_score_0_100, feature_contributions)
        """
        if not self._initialized:
            self.initialize()

        if self.session is None:
            # Dummy scoring for development
            return self._dummy_score(user_landmarks, user_visibility, trainer_landmarks, trainer_visibility)

        features = self.prepare_features(
            user_landmarks, user_visibility,
            trainer_landmarks, trainer_visibility,
            user_angles, trainer_angles
        )

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: features})[0]

        # Model outputs raw score, convert to 0-100
        raw_score = float(output[0][0]) if output.ndim > 1 else float(output[0])
        error_score = int(np.clip(raw_score * 100, 0, 100))

        # Feature importance (simplified)
        contributions = dict(zip(self._feature_names[:len(features[0])], features[0]))

        return error_score, contributions

    def _dummy_score(
        self,
        user_landmarks: np.ndarray,
        user_visibility: np.ndarray,
        trainer_landmarks: np.ndarray,
        trainer_visibility: np.ndarray,
    ) -> Tuple[int, Dict]:
        """Dummy scoring for development without trained model."""
        # Compute average landmark distance
        visible_mask = (user_visibility > 0.5) & (trainer_visibility > 0.5)
        if np.sum(visible_mask) < 5:
            return 50, {}

        distances = np.linalg.norm(
            user_landmarks[visible_mask] - trainer_landmarks[visible_mask],
            axis=1
        )
        avg_distance = np.mean(distances)

        # Convert to 0-100 score (lower distance = better form = lower error)
        error_score = int(np.clip(avg_distance * 100, 0, 100))

        return error_score, {}


class RuleBasedScorer:
    """Rule-based form scoring as fallback/interpretability layer."""

    EXERCISE_RULES = {
        "Squat": {
            "key_angles": ["left_knee", "right_knee", "left_hip", "right_hip", "torso_lean"],
            "thresholds": {
                "knee_min": 70,  # Minimum knee flexion
                "knee_max": 140,  # Maximum knee extension
                "hip_min": 60,
                "torso_max_lean": 45,
            }
        },
        "Deadlift": {
            "key_angles": ["left_hip", "right_hip", "left_knee", "right_knee", "torso_lean"],
            "thresholds": {
                "hip_min": 60,
                "hip_max": 160,
                "torso_max_lean": 90,
            }
        },
        "Bench Press": {
            "key_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
            "thresholds": {
                "elbow_min": 45,
                "elbow_max": 160,
                "shoulder_max": 90,
            }
        },
        # Add more exercises as needed
    }

    def __init__(self):
        pass

    def score_frame(
        self,
        exercise_name: str,
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
    ) -> Tuple[int, List[Dict]]:
        """Score frame using exercise-specific rules.

        Returns:
            Tuple of (error_score, list_of_violations)
        """
        rules = self.EXERCISE_RULES.get(exercise_name, {})
        thresholds = rules.get("thresholds", {})
        key_angles = rules.get("key_angles", [])

        violations = []
        total_penalty = 0

        for angle_name in key_angles:
            user_val = user_angles.get(angle_name)
            trainer_val = trainer_angles.get(angle_name)

            if user_val is None:
                continue

            # Compare to trainer
            diff = abs(user_val - (trainer_val or user_val))

            # Check thresholds
            penalty = 0
            if "knee_min" in thresholds and angle_name in ["left_knee", "right_knee"]:
                if user_val < thresholds["knee_min"]:
                    penalty = thresholds["knee_min"] - user_val
                    violations.append({
                        "joint": angle_name,
                        "issue": "Insufficient depth",
                        "user_angle": user_val,
                        "target_min": thresholds["knee_min"],
                        "severity": min(penalty / 20, 1.0)
                    })

            if "knee_max" in thresholds and angle_name in ["left_knee", "right_knee"]:
                if user_val > thresholds["knee_max"]:
                    penalty = user_val - thresholds["knee_max"]
                    violations.append({
                        "joint": angle_name,
                        "issue": "Hyperextension",
                        "user_angle": user_val,
                        "target_max": thresholds["knee_max"],
                        "severity": min(penalty / 20, 1.0)
                    })

            if "hip_min" in thresholds and angle_name in ["left_hip", "right_hip"]:
                if user_val < thresholds["hip_min"]:
                    penalty = thresholds["hip_min"] - user_val
                    violations.append({
                        "joint": angle_name,
                        "issue": "Insufficient hip hinge",
                        "user_angle": user_val,
                        "target_min": thresholds["hip_min"],
                        "severity": min(penalty / 20, 1.0)
                    })

            if "torso_max_lean" in thresholds and angle_name == "torso_lean":
                if user_val > thresholds["torso_max_lean"]:
                    penalty = user_val - thresholds["torso_max_lean"]
                    violations.append({
                        "joint": angle_name,
                        "issue": "Excessive forward lean",
                        "user_angle": user_val,
                        "target_max": thresholds["torso_max_lean"],
                        "severity": min(penalty / 30, 1.0)
                    })

            # General angle difference penalty
            diff_penalty = min(diff / 10, 5.0)
            total_penalty += max(penalty, diff_penalty)

        # Convert to 0-100 score
        error_score = int(np.clip(total_penalty * 10, 0, 100))

        return error_score, violations


def create_scorer(model_path: Optional[str] = None) -> FormScorer:
    """Factory function to create scorer."""
    return FormScorer(model_path)