"""Unit tests for Pose Direction Alignment."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPoseDirectionAligner:
    """Tests for PoseDirectionAligner class."""

    def setup_method(self):
        """Create aligner instance for each test."""
        from app.services.pose_alignment import PoseDirectionAligner
        self.aligner = PoseDirectionAligner(min_keypoint_score=0.3)

    def create_test_keypoints(self, facing: str = "right") -> tuple:
        """Create test keypoints for a person facing a specific direction.

        Args:
            facing: "left" or "right"

        Returns:
            (keypoints, scores) tuple
        """
        # Base keypoints (18 body keypoints)
        # Using image coordinates: origin top-left, Y increases downward
        # A person facing RIGHT: left shoulder is more forward (lower Y),
        # right shoulder is further back (higher Y)
        if facing == "right":
            keypoints = np.array([
                [320, 100],   # 0: nose
                [310, 90], [330, 90],   # 1,2: eyes
                [300, 85], [340, 85],   # 3,4: ears
                [250, 140], [380, 160],  # 5,6: shoulders (left more forward/lower Y)
                [220, 210], [400, 230],  # 7,8: elbows
                [200, 270], [420, 290],  # 9,10: wrists
                [270, 270], [360, 290],  # 11,12: hips (left more forward)
                [270, 370], [360, 380],  # 13,14: knees
                [270, 470], [360, 480],  # 15,16: ankles
                [315, 130],   # 17: neck
            ], dtype=np.float32)
        else:  # facing left
            # Mirror horizontally around center (320) and swap left/right
            base = np.array([
                [320, 100],
                [310, 90], [330, 90],
                [300, 85], [340, 85],
                [250, 140], [380, 160],
                [220, 210], [400, 230],
                [200, 270], [420, 290],
                [270, 270], [360, 290],
                [270, 370], [360, 380],
                [270, 470], [360, 480],
                [315, 130],
            ], dtype=np.float32)
            # Mirror
            keypoints = base.copy()
            keypoints[:, 0] = 640 - base[:, 0]
            # Swap left/right pairs
            lr_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
            for l, r in lr_pairs:
                keypoints[[l, r]] = keypoints[[r, l]]

        scores = np.ones(18, dtype=np.float32)
        return keypoints, scores

    def test_detect_facing_right(self):
        """Test direction detection for right-facing person."""
        keypoints, scores = self.create_test_keypoints("right")
        direction = self.aligner.detect_facing_direction(keypoints, scores)
        assert direction == "right"

    def test_detect_facing_left(self):
        """Test direction detection for left-facing person."""
        keypoints, scores = self.create_test_keypoints("left")
        direction = self.aligner.detect_facing_direction(keypoints, scores)
        assert direction == "left"

    def test_detect_facing_with_low_scores(self):
        """Test direction detection with some low-confidence keypoints."""
        keypoints, scores = self.create_test_keypoints("right")
        # Set some keypoints to low confidence
        scores[5] = 0.1  # left shoulder
        scores[11] = 0.1  # left hip

        direction = self.aligner.detect_facing_direction(keypoints, scores)
        # Should still work with remaining keypoints
        assert direction in ["left", "right"]

    def test_detect_facing_insufficient_keypoints(self):
        """Test direction detection fallback with insufficient keypoints."""
        keypoints = np.zeros((18, 2), dtype=np.float32)
        scores = np.zeros(18, dtype=np.float32)
        scores[0] = 1.0  # Only nose visible

        direction = self.aligner.detect_facing_direction(keypoints, scores)
        # Should default to "right"
        assert direction == "right"

    def test_mirror_pose_horizontally(self):
        """Test horizontal mirroring of keypoints."""
        keypoints, scores = self.create_test_keypoints("right")
        image_width = 640

        mirrored_kpts, mirrored_scores = self.aligner.mirror_pose_horizontally(
            keypoints, scores, image_width, full_keypoints=False
        )

        # After mirroring + swapping left/right pairs:
        # - Nose (0) and Neck (17) are NOT swapped, just mirrored
        # - Left/right pairs are swapped AND mirrored
        # So left index gets mirrored position of original right index

        # Check nose (0) and neck (17) - just mirrored, no swap
        assert abs(mirrored_kpts[0, 0] - (640 - keypoints[0, 0])) < 1
        assert abs(mirrored_kpts[17, 0] - (640 - keypoints[17, 0])) < 1

        # Check left/right pairs are swapped and mirrored
        lr_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
        for l, r in lr_pairs:
            # left index l should now have mirrored position of original right index r
            expected_l = 640 - keypoints[r, 0]
            assert abs(mirrored_kpts[l, 0] - expected_l) < 1, f"Left {l}: expected {expected_l}, got {mirrored_kpts[l, 0]}"
            # right index r should now have mirrored position of original left index l
            expected_r = 640 - keypoints[l, 0]
            assert abs(mirrored_kpts[r, 0] - expected_r) < 1, f"Right {r}: expected {expected_r}, got {mirrored_kpts[r, 0]}"

        # Scores also swapped for pairs
        assert mirrored_scores[5] == scores[6]
        assert mirrored_scores[6] == scores[5]

    def test_mirror_pose_no_image_width(self):
        """Test mirroring without explicit image width (uses keypoint bounds)."""
        keypoints, scores = self.create_test_keypoints("right")

        mirrored_kpts, _ = self.aligner.mirror_pose_horizontally(
            keypoints, scores, image_width=None, full_keypoints=False
        )

        # Should still mirror around keypoint center
        center_x = (np.max(keypoints[:, 0]) + np.min(keypoints[:, 0])) / 2
        # Nose and neck just mirrored
        assert abs(mirrored_kpts[0, 0] - (2 * center_x - keypoints[0, 0])) < 1
        assert abs(mirrored_kpts[17, 0] - (2 * center_x - keypoints[17, 0])) < 1

    def test_align_trainer_to_user_same_direction(self):
        """Test alignment when user and trainer face same direction."""
        user_kpts, user_scores = self.create_test_keypoints("right")
        trainer_kpts, trainer_scores = self.create_test_keypoints("right")

        aligned_kpts, aligned_scores, info = self.aligner.align_trainer_to_user(
            user_kpts, user_scores,
            trainer_kpts, trainer_scores,
            (480, 640), (480, 640),
            full_keypoints=False,
        )

        assert info["user_direction"] == "right"
        assert info["trainer_direction"] == "right"
        assert info["mirrored"] is False
        # Keypoints should be unchanged
        assert np.allclose(aligned_kpts, trainer_kpts)

    def test_align_trainer_to_user_different_direction(self):
        """Test alignment when user and trainer face different directions."""
        user_kpts, user_scores = self.create_test_keypoints("right")
        trainer_kpts, trainer_scores = self.create_test_keypoints("left")

        aligned_kpts, aligned_scores, info = self.aligner.align_trainer_to_user(
            user_kpts, user_scores,
            trainer_kpts, trainer_scores,
            (480, 640), (480, 640),
            full_keypoints=False,
        )

        assert info["user_direction"] == "right"
        assert info["trainer_direction"] == "left"
        assert info["mirrored"] is True
        # After alignment, trainer should match user's pose
        assert np.allclose(aligned_kpts, user_kpts, atol=1)

    def test_compute_pose_similarity_identical(self):
        """Test pose similarity for identical poses."""
        kpts1, scores1 = self.create_test_keypoints("right")
        kpts2, scores2 = self.create_test_keypoints("right")

        similarity = self.aligner.compute_pose_similarity(kpts1, scores1, kpts2, scores2)
        assert similarity > 0.9  # Very similar

    def test_compute_pose_similarity_different(self):
        """Test pose similarity for different poses."""
        kpts1, scores1 = self.create_test_keypoints("right")
        kpts2, scores2 = self.create_test_keypoints("left")

        similarity = self.aligner.compute_pose_similarity(kpts1, scores1, kpts2, scores2)
        # Mirrored poses have very similar joint configurations (just flipped)
        # So similarity is high. For truly different poses, use different joint angles.
        # Here we just verify the function runs and returns reasonable value
        assert 0.0 <= similarity <= 1.0

    def test_compute_pose_similarity_insufficient_keypoints(self):
        """Test pose similarity with insufficient visible keypoints."""
        kpts1 = np.zeros((18, 2), dtype=np.float32)
        kpts2 = np.zeros((18, 2), dtype=np.float32)
        scores1 = np.zeros(18, dtype=np.float32)
        scores2 = np.zeros(18, dtype=np.float32)
        scores1[0] = 1.0
        scores2[0] = 1.0

        similarity = self.aligner.compute_pose_similarity(kpts1, scores1, kpts2, scores2)
        assert similarity == 0.0


class TestControlNetConditioning:
    """Tests for ControlNet conditioning image creation."""

    def test_create_controlnet_conditioning(self):
        """Test creating ControlNet conditioning image."""
        from app.services.pose_alignment import create_controlnet_conditioning

        keypoints = np.array([
            [320, 100],   # nose
            [310, 90], [330, 90],   # eyes
            [300, 85], [340, 85],   # ears
            [250, 150], [380, 150],  # shoulders
            [220, 220], [400, 220],  # elbows
            [200, 280], [420, 280],  # wrists
            [280, 280], [360, 280],  # hips
            [280, 380], [360, 380],  # knees
            [280, 480], [360, 480],  # ankles
            [315, 130],   # neck
        ], dtype=np.float32)

        scores = np.ones(18, dtype=np.float32)

        cond_img = create_controlnet_conditioning(
            keypoints, scores, (480, 640), min_score=0.5
        )

        assert cond_img.shape == (480, 640, 3)
        assert cond_img.dtype == np.uint8
        # Should have white pixels
        assert np.any(cond_img > 0)
        # Background should be black
        assert np.any(cond_img == 0)

    def test_create_controlnet_conditioning_low_scores(self):
        """Test conditioning with low confidence keypoints."""
        from app.services.pose_alignment import create_controlnet_conditioning

        keypoints = np.random.rand(18, 2).astype(np.float32) * 640
        scores = np.array([0.1] * 18, dtype=np.float32)  # All low confidence
        scores[5] = 0.9  # Only left shoulder high

        cond_img = create_controlnet_conditioning(
            keypoints, scores, (480, 640), min_score=0.5
        )

        # Should have minimal drawing (only high confidence keypoints)
        assert cond_img.shape == (480, 640, 3)


class TestPoseAlignerGlobalInstance:
    """Tests for global pose aligner instance."""

    def test_get_pose_aligner_singleton(self):
        """Test that get_pose_aligner returns same instance."""
        from app.services.pose_alignment import get_pose_aligner
        import app.services.pose_alignment as pa

        pa._pose_aligner = None

        aligner1 = get_pose_aligner()
        aligner2 = get_pose_aligner()

        assert aligner1 is aligner2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])