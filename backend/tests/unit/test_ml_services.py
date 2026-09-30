"""Unit tests for DTW alignment helpers.

Pose decoding lives in test_pose_estimator.py, geometry in test_pose_features.py,
scoring/rules/rep counting in test_form_scoring.py, and the whole pipeline in
test_ml_pipeline.py.
"""

import numpy as np

from app.services.dtw_aligner import DTWAligner, compute_angle_differences
from app.services.pose_features import pose_embedding
from tests.synthetic_pose import VISIBLE, knee_cycle, sequence


class TestDTWAligner:
    """Tests for DTWAligner service."""

    def test_dtw_aligner_initialization(self):
        assert DTWAligner() is not None

    def test_compute_angle_differences(self):
        user_angles = {"left_knee": 90.0, "right_knee": 95.0, "left_hip": 110.0, "right_hip": 115.0}
        trainer_angles = {"left_knee": 95.0, "right_knee": 93.0, "left_hip": 115.0, "right_hip": 112.0}

        diffs = compute_angle_differences(user_angles, trainer_angles)

        assert diffs == {"left_knee": 5.0, "right_knee": 2.0, "left_hip": 5.0, "right_hip": 3.0}

    @staticmethod
    def embeddings(knees):
        return np.array([pose_embedding(p, VISIBLE) for p in sequence(knees)])

    def test_identical_sequences_align_diagonally_with_zero_distance(self):
        emb = self.embeddings(knee_cycle(2, 30, bottom=80.0))
        path, distance = DTWAligner(pose_estimator=object()).align_sequences(emb, emb)
        assert distance < 1e-6
        assert all(u == t for u, t in path)

    def test_slower_trainer_is_warped_onto_the_user(self):
        user = self.embeddings(knee_cycle(2, 20, bottom=80.0))
        trainer = self.embeddings(knee_cycle(2, 40, bottom=80.0))
        path, distance = DTWAligner(pose_estimator=object()).align_sequences(user, trainer)
        assert path[0] == (0, 0) and path[-1] == (len(user) - 1, len(trainer) - 1)
        assert len(path) >= len(trainer)
        assert distance < 0.1
