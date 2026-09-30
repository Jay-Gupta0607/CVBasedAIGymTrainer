"""Tests for joint angles and pose normalisation."""

import numpy as np
import pytest

from app.services.pose_features import (
    compute_angle_differences,
    compute_joint_angles,
    normalize_pose,
    pose_embedding,
    torso_length,
)
from tests.synthetic_pose import VISIBLE, squat_skeleton


class TestJointAngles:
    @pytest.mark.parametrize("knee", [180.0, 150.0, 120.0, 90.0, 70.0])
    def test_knee_angle_matches_construction(self, knee):
        angles = compute_joint_angles(squat_skeleton(knee))
        assert angles["left_knee"] == pytest.approx(knee, abs=0.5)
        assert angles["right_knee"] == pytest.approx(knee, abs=0.5)

    @pytest.mark.parametrize("elbow", [180.0, 120.0, 60.0])
    def test_elbow_angle_matches_construction(self, elbow):
        angles = compute_joint_angles(squat_skeleton(175.0, elbow_angle=elbow))
        assert angles["left_elbow"] == pytest.approx(elbow, abs=0.5)

    def test_angles_do_not_depend_on_scale_or_position(self):
        a = compute_joint_angles(squat_skeleton(100.0))
        b = compute_joint_angles(squat_skeleton(100.0, scale=700.0, offset=(320.0, 90.0)))
        for joint in a:
            assert b[joint] == pytest.approx(a[joint], abs=0.05)

    @pytest.mark.parametrize("lean", [0.0, 20.0, 45.0, 80.0])
    def test_torso_lean_is_angle_from_vertical(self, lean):
        angles = compute_joint_angles(squat_skeleton(175.0, torso_lean=lean))
        assert angles["torso_lean"] == pytest.approx(lean, abs=0.5)

    def test_upright_torso_is_not_reported_as_leaning(self):
        """Regression: torso_lean used to be shoulder-hip-hip (~90 deg when upright)."""
        assert compute_joint_angles(squat_skeleton(175.0, torso_lean=0.0))["torso_lean"] < 1.0

    def test_low_visibility_joints_are_omitted(self):
        visibility = VISIBLE.copy()
        visibility[27] = 0.1  # left ankle
        angles = compute_joint_angles(squat_skeleton(100.0), visibility)
        assert "left_knee" not in angles  # needs the ankle
        assert "right_knee" in angles
        assert "torso_lean" in angles

    def test_hidden_torso_omits_torso_lean(self):
        visibility = VISIBLE.copy()
        visibility[[11, 12]] = 0.0
        assert "torso_lean" not in compute_joint_angles(squat_skeleton(100.0), visibility)

    def test_no_visibility_means_all_angles(self):
        angles = compute_joint_angles(squat_skeleton(100.0))
        assert {"left_knee", "right_knee", "left_hip", "right_hip", "left_elbow", "torso_lean"} <= set(angles)


class TestAngleDifferences:
    def test_only_joints_measured_in_both(self):
        diffs = compute_angle_differences(
            {"left_knee": 90.0, "left_hip": 100.0}, {"left_knee": 95.0, "right_knee": 93.0}
        )
        assert diffs == {"left_knee": 5.0}


class TestNormalisation:
    def test_invariant_to_translation_and_scale(self):
        base = normalize_pose(squat_skeleton(100.0), VISIBLE)
        moved = normalize_pose(squat_skeleton(100.0, scale=350.0, offset=(500.0, -40.0)), VISIBLE)
        np.testing.assert_allclose(base, moved, atol=1e-4)

    def test_hip_centred_and_torso_unit_length(self):
        norm = normalize_pose(squat_skeleton(120.0, scale=200.0, offset=(10.0, 20.0)), VISIBLE)
        assert np.allclose(norm[[23, 24]].mean(axis=0), 0.0, atol=1e-5)
        assert torso_length(norm) == pytest.approx(1.0, abs=1e-4)

    def test_none_when_torso_not_visible(self):
        visibility = VISIBLE.copy()
        visibility[23] = 0.2
        assert normalize_pose(squat_skeleton(120.0), visibility) is None

    def test_none_for_degenerate_pose(self):
        assert normalize_pose(np.zeros((33, 3), dtype=np.float32), VISIBLE) is None

    def test_embedding_shape_and_missing_pose(self):
        assert pose_embedding(squat_skeleton(120.0), VISIBLE).shape == (99,)
        hidden = np.zeros(33, dtype=np.float32)
        assert not pose_embedding(squat_skeleton(120.0), hidden).any()

    def test_embedding_is_framing_invariant(self):
        a = pose_embedding(squat_skeleton(100.0), VISIBLE)
        b = pose_embedding(squat_skeleton(100.0, scale=900.0, offset=(100.0, 100.0)), VISIBLE)
        np.testing.assert_allclose(a, b, atol=1e-4)
