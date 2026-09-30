"""Tests for exercise rules, the landmark fallback scorer, the rule scorer, and rep counting."""

import logging

import numpy as np
import pytest

from app.models import ExerciseType
from app.services.dtw_aligner import DTWAligner
from app.services.exercise_rules import EXERCISE_PROFILES, SIDED_JOINTS, get_profile, joint_value
from app.services.form_scorer import FormScorer, RuleBasedScorer, UNKNOWN_SCORE
from app.services.pose_features import compute_joint_angles
from tests.synthetic_pose import VISIBLE, knee_cycle, squat_skeleton


# --------------------------------------------------------------------------- rules data
class TestExerciseRules:
    def test_every_exercise_has_a_profile(self):
        assert set(EXERCISE_PROFILES) == set(ExerciseType)

    def test_profiles_only_reference_known_joints_and_extremes(self):
        known = set(SIDED_JOINTS) | {"torso_lean"}
        for exercise, profile in EXERCISE_PROFILES.items():
            for joint in profile.primary:
                assert joint in known, (exercise, joint)
            for target in profile.rom:
                assert target.joint in known and target.extreme in ("min", "max"), exercise
                assert 0 < target.threshold < 180 and target.issue, exercise
            for rule in profile.limits:
                assert rule.joint in known, exercise
                assert rule.low is not None or rule.high is not None, exercise

    def test_lookup_accepts_common_spellings_and_unknown_is_empty(self):
        assert get_profile("squat") is get_profile("Squat")
        assert get_profile("push_up") is EXERCISE_PROFILES[ExerciseType.PUSH_UP]
        assert get_profile("burpee").primary == ()

    def test_joint_value_averages_measured_sides(self):
        assert joint_value({"left_knee": 90.0, "right_knee": 110.0}, "knee") == 100.0
        assert joint_value({"left_knee": 90.0}, "knee") == 90.0
        assert joint_value({}, "knee") is None
        assert joint_value({"torso_lean": 12.0}, "torso_lean") == 12.0


# ---------------------------------------------------------------- landmark fallback scorer
@pytest.fixture
def scorer_no_model():
    scorer = FormScorer(model_path="does-not-exist.onnx")
    scorer.initialize()
    return scorer


def fallback(scorer, user, trainer, uvis=VISIBLE, tvis=VISIBLE):
    return scorer.score_frame(user, uvis, trainer, tvis, {}, {})[0]


class TestFallbackScorer:
    def test_identical_poses_score_zero(self, scorer_no_model):
        pose = squat_skeleton(100.0)
        assert fallback(scorer_no_model, pose, pose) == 0

    def test_does_not_saturate_for_pixel_coordinates(self, scorer_no_model):
        """Regression: distance in pixels * 100, clipped to 100, made every frame 100."""
        user = squat_skeleton(100.0, scale=600.0, offset=(300.0, 200.0))
        trainer = squat_skeleton(105.0, scale=600.0, offset=(100.0, 250.0))
        assert fallback(scorer_no_model, user, trainer) < 25

    def test_same_pose_different_framing_scores_zero(self, scorer_no_model):
        user = squat_skeleton(100.0, scale=1000.0, offset=(640.0, 100.0))
        trainer = squat_skeleton(100.0, scale=300.0, offset=(20.0, 500.0))
        assert fallback(scorer_no_model, user, trainer) == 0

    def test_score_grows_with_pose_difference(self, scorer_no_model):
        trainer = squat_skeleton(70.0)
        scores = [fallback(scorer_no_model, squat_skeleton(k), trainer) for k in (70.0, 90.0, 120.0, 175.0)]
        assert scores == sorted(scores)
        assert scores[0] == 0 and scores[-1] > 50

    def test_scores_stay_within_bounds(self, scorer_no_model):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.uniform(0, 1000, size=(33, 3)).astype(np.float32)
            b = rng.uniform(0, 1000, size=(33, 3)).astype(np.float32)
            assert 0 <= fallback(scorer_no_model, a, b) <= 100

    def test_unmeasurable_pose_is_unknown_not_perfect_or_terrible(self, scorer_no_model):
        hidden = np.zeros(33, dtype=np.float32)
        assert fallback(scorer_no_model, squat_skeleton(100.0), squat_skeleton(100.0), uvis=hidden) == UNKNOWN_SCORE

    def test_feature_width_mismatch_falls_back_and_warns_once(self, scorer_no_model, caplog):
        class FakeSession:
            def run(self, *_):
                raise AssertionError("must not run with a mismatched feature width")

        scorer_no_model.session = FakeSession()
        scorer_no_model.input_name, scorer_no_model.output_name = "in", "out"
        scorer_no_model.expected_features = 156  # the shipped sample model

        pose = squat_skeleton(100.0)
        with caplog.at_level(logging.WARNING, logger="app.services.form_scorer"):
            for _ in range(5):
                score, _ = scorer_no_model.score_frame(pose, VISIBLE, pose, VISIBLE, {}, {})
        assert score == 0
        assert sum("feature mismatch" in r.message for r in caplog.records) == 1

    def test_feature_layout_is_273_and_framing_invariant(self, scorer_no_model):
        pose_a = squat_skeleton(100.0)
        pose_b = squat_skeleton(100.0, scale=500.0, offset=(50.0, 60.0))
        angles = compute_joint_angles(pose_a, VISIBLE)
        args = (VISIBLE, squat_skeleton(120.0), VISIBLE, angles, angles)
        fa = scorer_no_model.prepare_features(pose_a, *args)
        fb = scorer_no_model.prepare_features(pose_b, *args)
        assert fa.shape == (1, 273)
        np.testing.assert_allclose(fa, fb, atol=1e-4)


# ------------------------------------------------------------------------ rule scorer
def angles_for(knee, lean=15.0):
    return compute_joint_angles(squat_skeleton(knee, torso_lean=lean), VISIBLE)


class TestRuleBasedScorer:
    def test_matching_the_trainer_scores_zero_with_no_corrections(self):
        a = angles_for(100.0)
        score, violations = RuleBasedScorer().score_frame("Squat", a, a)
        assert score == 0 and violations == []

    def test_standing_is_not_flagged_as_hyperextension_or_lean(self):
        """Regression: static thresholds flagged every standing frame in a squat."""
        _, violations = RuleBasedScorer().score_frame("Squat", angles_for(178.0, lean=5.0), {})
        assert violations == []

    def test_deviation_from_trainer_is_reported_with_direction(self):
        score, violations = RuleBasedScorer().score_frame("Squat", angles_for(140.0), angles_for(90.0))
        assert score > 30
        assert any(v["joint"] == "knee" and "straighter" in v["issue"] for v in violations)

        _, violations = RuleBasedScorer().score_frame("Squat", angles_for(70.0), angles_for(110.0))
        assert any(v["joint"] == "knee" and "more bent" in v["issue"] for v in violations)

    def test_larger_deviation_scores_worse(self):
        trainer = angles_for(90.0)
        scores = [RuleBasedScorer().score_frame("Squat", angles_for(k), trainer)[0] for k in (90.0, 110.0, 130.0, 160.0)]
        assert scores == sorted(scores) and scores[-1] > scores[0]

    def test_torso_lean_limit_without_a_trainer(self):
        _, violations = RuleBasedScorer().score_frame("Squat", angles_for(100.0, lean=75.0), {})
        assert [v["issue"] for v in violations] == ["Excessive forward lean"]
        assert violations[0]["target_max"] == 60

    def test_unmeasurable_frame_is_unknown(self):
        assert RuleBasedScorer().score_frame("Squat", {}, {}) == (UNKNOWN_SCORE, [])

    def test_score_is_bounded_and_violations_sorted(self):
        score, violations = RuleBasedScorer().score_frame("Squat", angles_for(175.0, lean=90.0), angles_for(60.0))
        assert 0 <= score <= 100
        severities = [v["severity"] for v in violations]
        assert severities == sorted(severities, reverse=True)

    @pytest.mark.parametrize("exercise", list(ExerciseType))
    def test_every_exercise_scores_a_frame_and_a_rep(self, exercise):
        angles = angles_for(100.0)
        score, _ = RuleBasedScorer().score_frame(exercise.value, angles, angles_for(120.0))
        assert 0 <= score <= 100
        RuleBasedScorer().check_repetition(exercise.value, [angles] * 10)

    def test_unknown_exercise_still_compares_to_trainer(self):
        score, _ = RuleBasedScorer().score_frame("burpee", angles_for(60.0), angles_for(170.0))
        assert score > 0


class TestRepetitionRom:
    @staticmethod
    def rep(bottom, top=175.0, frames=30):
        return [angles_for(k) for k in knee_cycle(1, frames, bottom, top)]

    def test_deep_squat_passes(self):
        assert RuleBasedScorer().check_repetition("Squat", self.rep(bottom=85.0)) == []

    def test_shallow_squat_flags_depth(self):
        violations = RuleBasedScorer().check_repetition("Squat", self.rep(bottom=130.0))
        assert [v["issue"] for v in violations] == ["Insufficient depth"]
        assert violations[0]["target_angle"] == 100
        assert 0 < violations[0]["severity"] <= 1

    def test_incomplete_lockout_is_flagged(self):
        violations = RuleBasedScorer().check_repetition("Squat", self.rep(bottom=85.0, top=140.0))
        assert "Not standing fully at the top" in [v["issue"] for v in violations]

    def test_one_noisy_frame_does_not_pass_a_shallow_rep(self):
        frames = self.rep(bottom=130.0)
        frames[10] = angles_for(60.0)  # single glitch far below the real depth
        assert any(v["issue"] == "Insufficient depth" for v in RuleBasedScorer().check_repetition("Squat", frames))

    def test_too_few_measured_frames_is_not_judged(self):
        assert RuleBasedScorer().check_repetition("Squat", [angles_for(150.0)] * 3) == []

    def test_severity_scales_with_shortfall(self):
        mild = RuleBasedScorer().check_repetition("Squat", self.rep(bottom=110.0))[0]["severity"]
        bad = RuleBasedScorer().check_repetition("Squat", self.rep(bottom=140.0))[0]["severity"]
        assert 0 < mild < bad <= 1


# ------------------------------------------------------------------ rep segmentation
class TestRepSegmentation:
    aligner = DTWAligner(pose_estimator=object())

    @pytest.mark.parametrize("reps", [1, 3, 5, 8])
    def test_counts_each_rep_once(self, reps):
        signal = np.array(knee_cycle(reps, 30, bottom=80.0))
        segments = self.aligner.segment_repetitions_by_signal(signal, extreme="min")
        assert len(segments) == reps

    def test_segments_are_contiguous_and_cover_the_sequence(self):
        signal = np.array(knee_cycle(4, 30, bottom=80.0))
        segments = self.aligner.segment_repetitions_by_signal(signal, extreme="min")
        assert segments[0][0] == 0 and segments[-1][1] == len(signal) - 1
        for (_, end), (start, _) in zip(segments, segments[1:]):
            assert start == end
        for start, end in segments:  # each window holds its own bottom
            assert signal[start : end + 1].min() < 100

    def test_max_extreme_counts_lockouts(self):
        signal = 200.0 - np.array(knee_cycle(5, 30, bottom=80.0))  # flipped: reps are peaks
        assert len(self.aligner.segment_repetitions_by_signal(signal, extreme="max")) == 5

    def test_robust_to_noise_and_gaps(self):
        rng = np.random.default_rng(7)
        signal = np.array(knee_cycle(5, 30, bottom=80.0)) + rng.normal(0, 3, 150)
        signal[20:23] = np.nan
        signal[95] = np.nan
        assert len(self.aligner.segment_repetitions_by_signal(signal, extreme="min")) == 5

    def test_small_wobble_is_not_a_rep(self):
        signal = np.array(knee_cycle(6, 20, bottom=165.0, top=175.0))  # 10 degree swing
        assert self.aligner.segment_repetitions_by_signal(signal, extreme="min") == []

    def test_unmeasurable_signal_gives_no_reps(self):
        assert self.aligner.segment_repetitions_by_signal(np.full(50, np.nan)) == []
        assert self.aligner.segment_repetitions_by_signal(np.array([90.0, 100.0])) == []
