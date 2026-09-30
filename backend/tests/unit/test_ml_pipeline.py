"""End-to-end tests of MLPipeline on synthetic squats (fake pose estimator, video, storage).

These exercise the real DTW alignment, rep counting, scoring and summary code; only the
model and I/O are replaced. They show the logic is self-consistent - they do not show
that the ONNX pose model is accurate on real footage.
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.dtw_aligner import DTWAligner
from app.services.form_scorer import FormScorer, RuleBasedScorer
from app.services.ml_pipeline import MLPipeline
from tests.synthetic_pose import VISIBLE, knee_cycle, sequence


class ReplayPoseEstimator:
    """Returns pre-built skeletons; a frame's pixel [0,0] encodes (video id, frame index)."""

    def __init__(self, videos: Dict[int, List[np.ndarray]]):
        self.videos = videos

    def initialize(self):
        pass

    def estimate(self, frame):
        video, index = int(frame[0, 0, 0]), int(frame[0, 0, 1])
        return self.videos[video][index], VISIBLE.copy(), {}


class FakeVideoProcessor:
    def __init__(self, ids: Dict[str, int], lengths: Dict[str, int]):
        self.ids, self.lengths = ids, lengths

    def extract_frames(self, path):
        video = self.ids[path]
        frames = []
        for i in range(self.lengths[path]):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[0, 0] = (video, i, 0)
            frames.append(frame)
        return frames, {}

    def encode_frame_jpeg(self, frame):
        return b"jpeg"


def build_pipeline(videos: Dict[str, List[np.ndarray]]) -> MLPipeline:
    ids = {name: i for i, name in enumerate(videos)}
    estimator = ReplayPoseEstimator({ids[n]: poses for n, poses in videos.items()})
    scorer = FormScorer(model_path="does-not-exist.onnx")  # -> landmark-distance fallback
    with patch("app.services.ml_pipeline.get_storage_service", return_value=MagicMock()):
        return MLPipeline(
            pose_estimator=estimator,
            dwpose_estimator=MagicMock(),
            pose_aligner=MagicMock(),
            dtw_aligner=DTWAligner(estimator),
            form_scorer=scorer,
            rule_scorer=RuleBasedScorer(),
            video_processor=FakeVideoProcessor(ids, {n: len(p) for n, p in videos.items()}),
        )


def squats(bottom, reps=5, frames_per_rep=30, **kw):
    return sequence(knee_cycle(reps, frames_per_rep, bottom=bottom), **kw)


def issues(result) -> List[str]:
    return [d["title"] for d in result["technical_details"]]


class TestWithoutTrainer:
    def run(self, poses, exercise="Squat"):
        return build_pipeline({"user": poses}).analyze_without_trainer("user", exercise, task_id="t1")

    def test_counts_reps_and_passes_good_squats(self):
        result = self.run(squats(bottom=80.0, reps=5))
        assert result["reps"] == 5
        assert issues(result) == []
        assert "Completed 5 repetition(s)" in result["feedback_summary"]
        assert "excellent" in result["feedback_summary"]

    def test_shallow_squats_are_flagged_and_score_worse(self):
        good = self.run(squats(bottom=80.0))
        shallow = self.run(squats(bottom=135.0))
        assert "Insufficient depth" in issues(shallow)
        depth = next(d for d in shallow["technical_details"] if d["title"] == "Insufficient depth")
        assert "5 of 5 rep(s)" in depth["description"]
        assert "excellent" not in shallow["feedback_summary"]
        assert "excellent" in good["feedback_summary"]

    def test_only_the_bad_reps_are_attributed(self):
        knees = knee_cycle(3, 30, bottom=80.0) + knee_cycle(2, 30, bottom=135.0)
        result = self.run(sequence(knees))
        assert result["reps"] == 5
        depth = next(d for d in result["technical_details"] if d["title"] == "Insufficient depth")
        assert "2 of 5 rep(s)" in depth["description"]

    def test_excessive_lean_is_reported(self):
        result = self.run(squats(bottom=80.0, torso_lean=75.0))
        assert "Excessive forward lean" in issues(result)

    def test_result_shape(self):
        result = self.run(squats(bottom=80.0, reps=2))
        frame = result["analysis"][0]
        assert {"frame_id", "error_score", "feedback", "technical_observation",
                "user_image_key", "joint_angles"} <= set(frame)
        assert all(0 <= f["error_score"] <= 100 for f in result["analysis"])
        assert isinstance(result["technical_details"], list)

    def test_nothing_detected_says_so_instead_of_scoring(self):
        class Blind(ReplayPoseEstimator):
            def estimate(self, frame):
                landmarks, _, meta = super().estimate(frame)
                return landmarks, np.zeros(33, dtype=np.float32), meta

        pipeline = build_pipeline({"user": squats(bottom=80.0, reps=2)})
        pipeline.pose_estimator = Blind(pipeline.pose_estimator.videos)
        result = pipeline.analyze_without_trainer("user", "Squat", task_id="t1")
        assert "Could not reliably detect a person" in result["feedback_summary"]
        assert all(f["feedback"] == "Pose not clearly visible in this frame." for f in result["analysis"])

    def test_no_reps_means_no_grade(self):
        """A score from frames alone is meaningless if no repetition was found."""
        wobble = sequence(knee_cycle(6, 20, bottom=165.0, top=175.0))  # never really squats
        result = self.run(wobble)
        assert result["reps"] == 0
        assert "No complete repetitions" in result["feedback_summary"]
        assert "Overall form score" not in result["feedback_summary"]

    def test_exercise_without_rep_signal_still_completes(self):
        result = self.run(squats(bottom=80.0, reps=3), exercise="Plank")
        assert isinstance(result["reps"], int) and result["analysis"]


class TestWithTrainer:
    def run(self, user, trainer, exercise="Squat"):
        pipeline = build_pipeline({"user": user, "trainer": trainer})
        return pipeline.analyze_with_trainer("user", "trainer", exercise, task_id="t1")

    def test_matching_performance_scores_well(self):
        trainer = squats(bottom=80.0)
        user = squats(bottom=80.0, scale=400.0, offset=(120.0, 80.0))  # different framing/resolution
        result = self.run(user, trainer)
        assert result["reps"] == 5
        assert np.mean([f["error_score"] for f in result["analysis"]]) < 10
        assert "excellent" in result["feedback_summary"]

    def test_user_who_is_too_shallow_is_told_how(self):
        result = self.run(squats(bottom=135.0), squats(bottom=80.0))
        assert "Insufficient depth" in issues(result)
        assert "Knees straighter than the trainer" in issues(result)
        assert "excellent" not in result["feedback_summary"]

    def test_worse_form_gets_a_worse_score(self):
        """Score must rise monotonically with depth shortfall (85 -> 110 -> 140 degrees)."""
        trainer = squats(bottom=80.0)
        results = [self.run(squats(bottom=b), trainer) for b in (85.0, 110.0, 140.0)]
        errors = [np.array([f["error_score"] for f in r["analysis"]]) for r in results]

        means = [e.mean() for e in errors]
        assert means == sorted(means) and means[0] < 3 < means[2]
        # Standing frames match the trainer exactly and dilute the mean; the damage is in
        # the frames near the bottom of each rep.
        p90 = [np.percentile(e, 90) for e in errors]
        assert p90[2] > p90[0] + 30

    def test_summary_grade_follows_depth(self):
        trainer = squats(bottom=80.0)
        summaries = [self.run(squats(bottom=b), trainer)["feedback_summary"] for b in (85.0, 140.0)]
        assert "excellent" in summaries[0]
        assert "needs improvement" in summaries[1] or "poor" in summaries[1]

    def test_different_speed_is_aligned_by_dtw(self):
        """Trainer does slow reps, user fast reps of the same depth: should still match."""
        trainer = squats(bottom=80.0, reps=3, frames_per_rep=50)
        user = squats(bottom=80.0, reps=3, frames_per_rep=25)
        result = self.run(user, trainer)
        assert result["reps"] == 3
        assert np.mean([f["error_score"] for f in result["analysis"]]) < 15

    def test_violations_do_not_inflate_the_issue_count(self):
        """Regression: the summary used to count one 'issue' per violating frame."""
        result = self.run(squats(bottom=135.0), squats(bottom=80.0))
        distinct = int(result["feedback_summary"].split("Found ")[1].split(" distinct")[0])
        assert distinct <= 8
