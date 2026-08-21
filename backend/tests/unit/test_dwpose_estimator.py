"""Unit tests for DWPose Estimator."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Test imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDWPoseEstimator:
    """Tests for DWPoseEstimator class."""

    def test_initialization_without_model(self):
        """Test that initialization fails gracefully without model."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator(model_path="/nonexistent/path.onnx")
        assert not estimator.is_healthy()

        with pytest.raises(FileNotFoundError):
            estimator.initialize()

    @patch("app.services.dwpose_estimator.ort.InferenceSession")
    def test_initialization_with_mock_model(self, mock_session_class):
        """Test initialization with mocked ONNX session."""
        from app.services.dwpose_estimator import DWPoseEstimator

        # Setup mock
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.get_outputs.return_value = [
            MagicMock(name="keypoints"),
            MagicMock(name="scores"),
        ]
        mock_session_class.return_value = mock_session

        # Create temp file to satisfy Path.exists()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        try:
            estimator = DWPoseEstimator(model_path=temp_path)
            estimator.initialize()

            assert estimator.is_healthy()
            assert estimator.session is not None
            mock_session_class.assert_called_once()
        finally:
            import os
            os.unlink(temp_path)

    def test_preprocess_output_shape(self):
        """Test preprocessing output shape."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()
        estimator._initialized = True
        estimator.input_size = (288, 384)

        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        input_tensor, metadata = estimator.preprocess(frame)

        assert input_tensor.shape == (1, 3, 288, 384)
        assert metadata["original_shape"] == (480, 640)
        assert "scale" in metadata
        assert "padding" in metadata

    def test_postprocess_denormalization(self):
        """Test postprocessing correctly denormalizes keypoints."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()
        estimator.input_size = (288, 384)

        # Mock outputs: keypoints in input space, scores
        keypoints_normalized = np.random.rand(130, 2).astype(np.float32) * 384
        scores = np.ones(130, dtype=np.float32)
        outputs = [keypoints_normalized[np.newaxis, ...], scores[np.newaxis, ...]]

        metadata = {
            "original_shape": (480, 640),
            "scale": 0.6,
            "padding": (24, 48),
        }

        keypoints, scores_out = estimator.postprocess(outputs, metadata)

        assert keypoints.shape == (130, 2)
        assert scores_out.shape == (130,)
        # Check denormalization happened
        assert np.all(keypoints[:, 0] >= 0)
        assert np.all(keypoints[:, 0] <= 640)
        assert np.all(keypoints[:, 1] >= 0)
        assert np.all(keypoints[:, 1] <= 480)

    def test_get_body_keypoints(self):
        """Test extracting body keypoints from full set."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()

        full_kpts = np.random.rand(130, 2).astype(np.float32)
        full_scores = np.random.rand(130).astype(np.float32)

        body_kpts, body_scores = estimator.get_body_keypoints(full_kpts, full_scores)

        assert body_kpts.shape == (18, 2)
        assert body_scores.shape == (18,)

    def test_get_face_keypoints(self):
        """Test extracting face keypoints from full set."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()

        full_kpts = np.random.rand(130, 2).astype(np.float32)
        full_scores = np.random.rand(130).astype(np.float32)

        face_kpts, face_scores = estimator.get_face_keypoints(full_kpts, full_scores)

        assert face_kpts.shape == (68, 2)
        assert face_scores.shape == (68,)

    def test_get_hand_keypoints(self):
        """Test extracting hand keypoints from full set."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()

        full_kpts = np.random.rand(130, 2).astype(np.float32)
        full_scores = np.random.rand(130).astype(np.float32)

        left_hand, right_hand = estimator.get_hand_keypoints(full_kpts, full_scores)

        left_kpts, left_scores = left_hand
        right_kpts, right_scores = right_hand

        assert left_kpts.shape == (21, 2)
        assert left_scores.shape == (21,)
        assert right_kpts.shape == (21, 2)
        assert right_scores.shape == (21,)

    def test_create_controlnet_image(self):
        """Test ControlNet conditioning image creation."""
        from app.services.dwpose_estimator import DWPoseEstimator

        estimator = DWPoseEstimator()

        keypoints = np.array([
            [100, 100],  # nose
            [90, 90], [110, 90],  # eyes
            [80, 80], [120, 80],  # ears
            [80, 150], [120, 150],  # shoulders
            [60, 200], [140, 200],  # elbows
            [50, 250], [150, 250],  # wrists
            [90, 250], [110, 250],  # hips
            [90, 350], [110, 350],  # knees
            [90, 450], [110, 450],  # ankles
            [100, 130],  # neck
        ], dtype=np.float32)

        scores = np.ones(18, dtype=np.float32)

        control_img = estimator.create_controlnet_image(
            keypoints, scores, (480, 640), threshold=0.5
        )

        assert control_img.shape == (480, 640, 3)
        assert control_img.dtype == np.uint8
        # Should have white pixels where skeleton drawn
        assert np.any(control_img > 0)


class TestDWPoseGlobalInstance:
    """Tests for global DWPose instance management."""

    def test_get_dwpose_estimator_singleton(self):
        """Test that get_dwpose_estimator returns same instance."""
        from app.services.dwpose_estimator import get_dwpose_estimator, _dwpose_estimator

        # Reset global
        import app.services.dwpose_estimator as de
        de._dwpose_estimator = None

        est1 = get_dwpose_estimator()
        est2 = get_dwpose_estimator()

        assert est1 is est2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])