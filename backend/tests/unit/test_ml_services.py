"""Unit tests for ML services."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestPoseEstimator:
    """Tests for PoseEstimator service."""

    @pytest.mark.asyncio
    async def test_pose_estimator_initialization(self):
        """Test pose estimator initialization."""
        from app.services.pose_estimator import PoseEstimator

        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_session.return_value = MagicMock()
            estimator = PoseEstimator()
            estimator.initialize()
            assert estimator.session is not None

    @pytest.mark.asyncio
    async def test_pose_estimator_estimate(self):
        """Test pose estimation on a frame."""
        from app.services.pose_estimator import PoseEstimator

        with patch("onnxruntime.InferenceSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_session.get_outputs.return_value = [MagicMock(name="output")]
            # Mock output: (1, 33, 3) landmarks + (1, 33) confidence
            mock_session.run.return_value = [
                np.random.randn(1, 33, 3).astype(np.float32),
                np.random.rand(1, 33).astype(np.float32)
            ]
            mock_session_class.return_value = mock_session

            estimator = PoseEstimator()
            estimator.initialize()

            # Create dummy frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = estimator.estimate(frame)

            assert "landmarks" in result
            assert "confidence" in result
            assert len(result["landmarks"]) == 33

    @pytest.mark.asyncio
    async def test_pose_estimator_preprocess(self):
        """Test frame preprocessing."""
        from app.services.pose_estimator import PoseEstimator

        estimator = PoseEstimator()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = estimator.preprocess(frame)

        assert processed.shape == (1, 3, 256, 256)  # CHW format
        assert processed.dtype == np.float32


class TestDTWAligner:
    """Tests for DTWAligner service."""

    def test_dtw_aligner_initialization(self):
        """Test DTW aligner initialization."""
        from app.services.dtw_aligner import DTWAligner

        aligner = DTWAligner()
        assert aligner is not None

    def test_compute_joint_angles(self):
        """Test joint angle computation."""
        from app.services.dtw_aligner import compute_joint_angles

        # Create dummy landmarks (33, 3)
        landmarks = np.random.randn(33, 3).astype(np.float32)
        # Set some known positions for testing
        landmarks[11] = [0.4, 0.5, 0.0]  # Left shoulder
        landmarks[13] = [0.4, 0.6, 0.0]  # Left elbow
        landmarks[15] = [0.4, 0.7, 0.0]  # Left wrist
        landmarks[12] = [0.6, 0.5, 0.5, 0.0]  # Right shoulder
        landmarks[14] = [0.6, 0.6, 0.0]  # Right elbow
        landmarks[16] = [0.6, 0.7, 0.0]  # Right wrist
        landmarks[23] = [0.45, 0.8, 0.0]  # Left hip
        landmarks[25] = [0.45, 0.9, 0.0]  # Left knee
        landmarks[27] = [0.45, 1.0, 0.0]  # Left ankle
        landmarks[24] = [0.55, 0.8, 0.0]  # Right hip
        landmarks[26] = [0.55, 0.9, 0.0]  # Right knee
        landmarks[28] = [0.55, 1.0, 0.0]  # Right ankle

        angles = compute_joint_angles(landmarks)

        assert "left_elbow" in angles
        assert "right_elbow" in angles
        assert "left_knee" in angles
        assert "right_knee" in angles
        assert "left_hip" in angles
        assert "right_hip" in angles

    def test_compute_angle_differences(self):
        """Test angle difference computation."""
        from app.services.dtw_aligner import compute_angle_differences

        user_angles = {"left_knee": 90.0, "right_knee": 95.0, "left_hip": 110.0, "right_hip": 115.0}
        trainer_angles = {"left_knee": 95.0, "right_knee": 93.0, "left_hip": 115.0, "right_hip": 112.0}

        diffs = compute_angle_differences(user_angles, trainer_angles)

        assert "left_knee" in diffs
        assert "right_knee" in diffs
        assert "left_hip" in diffs
        assert "right_hip" in diffs
        assert diffs["left_knee"] == 5.0
        assert diffs["right_knee"] == 2.0


class TestFormScorer:
    """Tests for FormScorer service."""

    def test_rule_based_scorer_initialization(self):
        """Test rule-based scorer initialization."""
        from app.services.form_scorer import RuleBasedScorer

        scorer = RuleBasedScorer()
        assert scorer is not None
        assert hasattr(scorer, "exercise_rules")

    def test_rule_based_scorer_score_frame(self):
        """Test rule-based scoring."""
        from app.services.form_scorer import RuleBasedScorer

        scorer = RuleBasedScorer()

        user_landmarks = np.random.randn(33, 3).astype(np.float32)
        trainer_landmarks = np.random.randn(33, 3).astype(np.float32)
        user_angles = {"left_knee": 90.0, "right_knee": 95.0, "left_hip": 110.0, "right_hip": 115.0}
        trainer_angles = {"left_knee": 95.0, "right_knee": 93.0, "left_hip": 115.0, "right_hip": 112.0}

        result = scorer.score_frame(
            user_landmarks,
            trainer_landmarks,
            user_angles,
            trainer_angles,
            "squat"
        )

        assert "error_score" in result
        assert "feedback" in result
        assert "technical_observation" in result
        assert 0 <= result["error_score"] <= 100


class TestMLPipeline:
    """Tests for MLPipeline service."""

    @pytest.mark.asyncio
    async def test_ml_pipeline_initialization(self):
        """Test ML pipeline initialization."""
        from app.services.ml_pipeline import MLPipeline

        with patch("app.services.ml_pipeline.PoseEstimator") as mock_pose:
            with patch("app.services.ml_pipeline.DTWAligner") as mock_dtw:
                with patch("app.services.ml_pipeline.create_scorer") as mock_scorer:
                    mock_pose.return_value = MagicMock()
                    mock_dtw.return_value = MagicMock()
                    mock_scorer.return_value = MagicMock()

                    pipeline = MLPipeline()
                    assert pipeline.pose_estimator is not None
                    assert pipeline.dtw_aligner is not None
                    assert pipeline.scorer is not None

    @pytest.mark.asyncio
    async def test_extract_frames(self):
        """Test frame extraction from video."""
        from app.services.ml_pipeline import MLPipeline

        with patch("app.services.ml_pipeline.VideoProcessor") as mock_processor:
            mock_processor.return_value.extract_frames.return_value = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
            ]
            mock_processor.return_value.get_video_info.return_value = {
                "fps": 30, "duration": 1.0, "width": 640, "height": 480
            }

            pipeline = MLPipeline()
            frames = pipeline.extract_frames("dummy.mp4", fps=30)

            assert len(frames) == 10