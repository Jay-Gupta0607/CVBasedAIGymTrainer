"""Form scoring: optional LightGBM ONNX model, a scale-invariant fallback, and rules."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from app.config import settings
from app.services.exercise_rules import DEVIATION_TEXT, get_profile, joint_value
from app.services.pose_features import normalize_pose

logger = logging.getLogger(__name__)

# Body-core landmarks used to compare poses (shoulders, elbows, wrists, hips, knees,
# ankles). Face, hands and feet are noisy and irrelevant to lifting form.
CORE_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Fallback landmark scorer: mean per-landmark distance (in torso lengths) between two
# poses after normalising both. Calibrated on SYNTHETIC squat geometry, where standing vs
# the bottom of a squat - the largest difference that movement has - is ~0.25; people in
# the same pose still differ a little (proportions, estimation noise). Re-tune both
# numbers against real footage.
POSE_DISTANCE_NOISE_FLOOR = 0.05
POSE_DISTANCE_FULL_PENALTY = 0.30

# Score returned when a frame cannot be measured (pose not reliably visible)
UNKNOWN_SCORE = 50

ANGLE_FEATURES = [
    "left_elbow", "left_shoulder", "right_elbow", "right_shoulder",
    "left_knee", "left_hip", "right_knee", "right_hip", "torso_lean",
]


class FormScorer:
    """LightGBM-based form scoring using ONNX Runtime."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.SCORER_MODEL_PATH
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self._initialized = False
        self._feature_names = self._get_feature_names()
        self.expected_features: Optional[int] = None
        self._warned_mismatch = False

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
            logger.warning(f"Scorer model not found at {model_path}, using the landmark-distance fallback")
            self._initialized = True
            return

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = settings.onnx_providers_list
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

        # Record the model's expected feature width. The runtime feature layout
        # (273 features) diverged from the training layout used to build this ONNX
        # (156 generic col_* features), so a mismatch must fall back to the dummy
        # scorer rather than fail ONNX with an INVALID_ARGUMENT dimension error.
        dims = self.session.get_inputs()[0].shape
        self.expected_features = int(dims[1]) if len(dims) > 1 and dims[1] is not None else None

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
    ) -> Optional[np.ndarray]:
        """Prepare feature vector for scoring.

        Args:
            user_landmarks: User pose landmarks (33, 3)
            user_visibility: User landmark visibility (33,)
            trainer_landmarks: Trainer pose landmarks (33, 3)
            trainer_visibility: Trainer landmark visibility (33,)
            user_angles: User joint angles
            trainer_angles: Trainer joint angles

        Returns:
            Feature vector (1, n_features), or None if feature layout mismatches the model
        """
        # Landmarks are normalised (hip-centred, in torso lengths) so features do not
        # depend on framing or resolution. Poses without a visible torso cannot be
        # normalised -> None -> fallback.
        user_norm = normalize_pose(user_landmarks, user_visibility)
        trainer_norm = normalize_pose(trainer_landmarks, trainer_visibility)
        if user_norm is None or trainer_norm is None:
            return None

        features = []
        features.extend(user_norm.reshape(-1).tolist())
        features.extend(trainer_norm.reshape(-1).tolist())

        # Angle differences; NaN when either side was not measured (LightGBM treats NaN
        # as missing, unlike a fake 0 which would read as "identical")
        for angle in ANGLE_FEATURES:
            if angle in user_angles and angle in trainer_angles:
                features.append(abs(user_angles[angle] - trainer_angles[angle]))
            else:
                features.append(float("nan"))

        # Visibility
        features.extend(user_visibility.tolist())
        features.extend(trainer_visibility.tolist())

        vec = np.array(features, dtype=np.float32).reshape(1, -1)

        # Guard against model/runtime feature-layout drift: returning None signals
        # score_frame to use the fallback instead of a hard ONNX dimension error.
        if self.expected_features is not None and vec.shape[1] != self.expected_features:
            if not self._warned_mismatch:
                self._warned_mismatch = True
                logger.warning(
                    f"Form scorer feature mismatch: built {vec.shape[1]} features, "
                    f"model expects {self.expected_features}. Using the landmark-distance "
                    "fallback (retrain on this layout to use the ML scorer)."
                )
            return None

        return vec

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
            # No ML model loaded: use the landmark-distance fallback
            return self._landmark_distance_score(user_landmarks, user_visibility, trainer_landmarks, trainer_visibility)

        features = self.prepare_features(
            user_landmarks, user_visibility,
            trainer_landmarks, trainer_visibility,
            user_angles, trainer_angles
        )

        if features is None:
            return self._landmark_distance_score(user_landmarks, user_visibility, trainer_landmarks, trainer_visibility)

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: features})[0]

        # Model outputs raw score, convert to 0-100
        raw_score = float(output[0][0]) if output.ndim > 1 else float(output[0])
        error_score = int(np.clip(raw_score * 100, 0, 100))

        # Feature importance (simplified)
        contributions = dict(zip(self._feature_names[:len(features[0])], features[0]))

        return error_score, contributions

    def _landmark_distance_score(
        self,
        user_landmarks: np.ndarray,
        user_visibility: np.ndarray,
        trainer_landmarks: np.ndarray,
        trainer_visibility: np.ndarray,
    ) -> Tuple[int, Dict]:
        """Scale-invariant pose difference, 0 (same pose) to 100 (very different).

        Both poses are normalised to hip-centred torso lengths first, so the score does
        not depend on where the person stands, how far they are, or the resolution.
        """
        user_norm = normalize_pose(user_landmarks, user_visibility)
        trainer_norm = normalize_pose(trainer_landmarks, trainer_visibility)
        if user_norm is None or trainer_norm is None:
            return UNKNOWN_SCORE, {}

        core = np.array(CORE_LANDMARKS)
        visible = (user_visibility[core] >= 0.5) & (trainer_visibility[core] >= 0.5)
        if visible.sum() < 4:
            return UNKNOWN_SCORE, {}

        distances = np.linalg.norm(user_norm[core][visible] - trainer_norm[core][visible], axis=1)
        mean_distance = float(np.mean(distances))

        span = POSE_DISTANCE_FULL_PENALTY - POSE_DISTANCE_NOISE_FLOOR
        error = np.clip((mean_distance - POSE_DISTANCE_NOISE_FLOOR) / span, 0.0, 1.0)
        return int(round(error * 100)), {"mean_pose_distance_torso_lengths": mean_distance}


class RuleBasedScorer:
    """Rule-based scoring driven by the per-exercise profiles in `exercise_rules`.

    Two kinds of check:
      * per frame (`score_frame`): joint angles vs the trainer's (after alignment) and
        hard bounds such as torso lean;
      * per repetition (`check_repetition`): did the rep reach the target range of
        motion (depth, lockout, ...).
    """

    # Full comparison penalty when a joint is this many degrees off the trainer's
    COMPARE_FULL_PENALTY_DEG = 40.0
    # Deviations beyond this are reported as a correction
    COMPARE_VIOLATION_DEG = 15.0
    # Degrees past a hard limit / short of a ROM target for full severity
    LIMIT_FULL_SEVERITY_DEG = 20.0
    ROM_FULL_SEVERITY_DEG = 30.0
    # Ignore ROM checks on windows with fewer measured frames than this
    MIN_ROM_FRAMES = 5
    # A rep is judged on the 5th/95th percentile of its angle, not the raw extreme,
    # so one noisy frame cannot pass or fail a whole repetition
    ROM_PERCENTILE = 5

    def score_frame(
        self,
        exercise_name: str,
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
    ) -> Tuple[int, List[Dict]]:
        """Score one frame.

        Returns:
            (error_score 0-100, violations). A frame with no measurable joints scores
            UNKNOWN_SCORE with no violations. Without trainer angles only the hard
            limits contribute.
        """
        profile = get_profile(exercise_name)
        compare_joints = (
            profile.primary + ("torso_lean",)
            if profile.primary
            else ("elbow", "shoulder", "knee", "hip", "torso_lean")
        )

        measured = any(joint_value(user_angles, j) is not None for j in compare_joints)
        if not measured:
            return UNKNOWN_SCORE, []

        violations: List[Dict] = []
        comparison_terms: List[float] = []

        for joint in compare_joints:
            user_val = joint_value(user_angles, joint)
            trainer_val = joint_value(trainer_angles, joint) if trainer_angles else None
            if user_val is None or trainer_val is None:
                continue
            deviation = user_val - trainer_val
            severity = min(abs(deviation) / self.COMPARE_FULL_PENALTY_DEG, 1.0)
            comparison_terms.append(severity)
            if abs(deviation) > self.COMPARE_VIOLATION_DEG:
                smaller, larger = DEVIATION_TEXT[joint]
                violations.append({
                    "joint": joint,
                    "issue": smaller if deviation < 0 else larger,
                    "user_angle": user_val,
                    "target_angle": trainer_val,
                    "severity": severity,
                })

        limit_severity = 0.0
        for rule in profile.limits:
            value = joint_value(user_angles, rule.joint)
            if value is None:
                continue
            for bound, exceeded, issue, key in (
                (rule.low, rule.low is not None and value < rule.low, rule.issue_low, "target_min"),
                (rule.high, rule.high is not None and value > rule.high, rule.issue_high, "target_max"),
            ):
                if exceeded:
                    severity = min(abs(value - bound) / self.LIMIT_FULL_SEVERITY_DEG, 1.0)
                    limit_severity = max(limit_severity, severity)
                    violations.append({
                        "joint": rule.joint,
                        "issue": issue,
                        "user_angle": value,
                        key: bound,
                        "severity": severity,
                    })

        comparison = float(np.mean(comparison_terms)) if comparison_terms else 0.0
        error = min(1.0, comparison + 0.5 * limit_severity)
        violations.sort(key=lambda v: v["severity"], reverse=True)
        return int(round(error * 100)), violations

    def check_repetition(
        self,
        exercise_name: str,
        rep_angles: List[Dict[str, float]],
    ) -> List[Dict]:
        """Check one repetition's range of motion.

        Args:
            rep_angles: joint-angle dicts of the frames belonging to the repetition

        Returns:
            Violations for each ROM target the repetition missed.
        """
        violations: List[Dict] = []
        for target in get_profile(exercise_name).rom:
            values = [v for v in (joint_value(a, target.joint) for a in rep_angles) if v is not None]
            if len(values) < self.MIN_ROM_FRAMES:
                continue
            if target.extreme == "min":
                reached = float(np.percentile(values, self.ROM_PERCENTILE))
                shortfall = reached - target.threshold
            else:
                reached = float(np.percentile(values, 100 - self.ROM_PERCENTILE))
                shortfall = target.threshold - reached
            if shortfall > 0:
                violations.append({
                    "joint": target.joint,
                    "issue": target.issue,
                    "user_angle": reached,
                    "target_angle": target.threshold,
                    "severity": min(shortfall / self.ROM_FULL_SEVERITY_DEG, 1.0),
                })
        return violations


def create_scorer(model_path: Optional[str] = None) -> FormScorer:
    """Factory function to create scorer."""
    return FormScorer(model_path)