"""Unit tests for PoseEstimator pre/post-processing (no model file needed).

The decode math is checked by a round trip: known original-pixel positions are pushed
through the *forward* transform by hand to fabricate raw model output, and
`estimate()` must recover them. The same math was verified against MediaPipe's
official PoseLandmarker (see scripts/verify_pose_decode.py).
"""

import numpy as np
import pytest

from app.services.pose_estimator import (
    MODEL_INPUT_SIZE,
    PoseEstimator,
    letterbox_roi,
)


class FakeSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, raw_landmarks: np.ndarray):
        self.raw = raw_landmarks.reshape(1, -1).astype(np.float32)
        self.last_input = None

    def run(self, output_names, feed):
        self.last_input = next(iter(feed.values()))
        return [self.raw]


def make_estimator(raw_landmarks: np.ndarray) -> PoseEstimator:
    est = PoseEstimator(model_path="unused.onnx")
    est.session = FakeSession(raw_landmarks)
    est.input_name = "input"
    est.output_names = ["landmarks"]
    est._initialized = True
    return est


def raw_from_pixels(xy_z, vis_prob, roi):
    """Forward transform: original pixels -> raw 39x5 model output (256-space, logits)."""
    cx, cy, side = roi
    raw = np.zeros((39, 5), dtype=np.float32)
    k = MODEL_INPUT_SIZE / side
    raw[:33, 0] = (xy_z[:, 0] - (cx - side / 2)) * k
    raw[:33, 1] = (xy_z[:, 1] - (cy - side / 2)) * k
    raw[:33, 2] = xy_z[:, 2] * k
    p = np.clip(vis_prob, 1e-4, 1 - 1e-4)
    raw[:33, 3] = np.log(p / (1 - p))
    return raw


@pytest.fixture
def pose_pixels():
    rng = np.random.default_rng(0)
    return rng.uniform(0.15, 0.85, size=(33, 3)).astype(np.float32)


@pytest.mark.parametrize("shape", [(480, 640), (640, 480), (1080, 1920), (720, 720), (1920, 1080)])
def test_letterbox_round_trip_recovers_original_pixels(shape, pose_pixels):
    h, w = shape
    truth = pose_pixels * np.array([w, h, 100.0], dtype=np.float32)
    vis = np.linspace(0.05, 0.95, 33).astype(np.float32)

    est = make_estimator(raw_from_pixels(truth, vis, letterbox_roi(h, w)))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    landmarks, visibility, _ = est.estimate(frame)

    np.testing.assert_allclose(landmarks, truth, atol=1e-2)
    np.testing.assert_allclose(visibility, vis, atol=1e-3)


def test_custom_roi_round_trip_including_offscreen_roi(pose_pixels):
    h, w = 600, 512
    # ROI larger than the frame and centred below it (like a full-body crop of a portrait)
    roi = (297.0, 916.0, 1815.0)
    truth = pose_pixels * np.array([w, h, 100.0], dtype=np.float32)
    vis = np.full(33, 0.9, dtype=np.float32)

    est = make_estimator(raw_from_pixels(truth, vis, roi))
    landmarks, _, _ = est.estimate(np.zeros((h, w, 3), np.uint8), roi=roi)

    np.testing.assert_allclose(landmarks, truth, atol=5e-2)


def test_old_letterbox_bug_would_have_failed():
    """Non-square frame: x/y must be in original pixels, not 256-space."""
    h, w = 480, 640
    truth = np.zeros((33, 3), dtype=np.float32)
    truth[0] = [320.0, 240.0, 0.0]  # frame centre
    est = make_estimator(raw_from_pixels(truth, np.full(33, 0.9), letterbox_roi(h, w)))
    landmarks, _, _ = est.estimate(np.zeros((h, w, 3), np.uint8))
    assert landmarks[0, 0] == pytest.approx(320.0, abs=1e-2)
    assert landmarks[0, 1] == pytest.approx(240.0, abs=1e-2)


def test_offscreen_landmarks_are_not_clipped_to_the_frame():
    h, w = 480, 640
    truth = np.zeros((33, 3), dtype=np.float32)
    truth[27] = [300.0, 700.0, 0.0]  # ankle below the frame
    est = make_estimator(raw_from_pixels(truth, np.full(33, 0.9), letterbox_roi(h, w)))
    landmarks, _, _ = est.estimate(np.zeros((h, w, 3), np.uint8))
    assert landmarks[27, 1] == pytest.approx(700.0, abs=1e-2)


def test_z_is_scaled_like_xy_so_angles_do_not_depend_on_resolution():
    """Same pose at 2x resolution must give 2x coordinates in all three axes."""
    pose = np.array([[0.4, 0.5, 0.1]] * 33, dtype=np.float32)
    outs = []
    for scale in (1, 2):
        h, w = 480 * scale, 640 * scale
        truth = pose * np.array([w, h, 640.0 * scale], dtype=np.float32)
        est = make_estimator(raw_from_pixels(truth, np.full(33, 0.9), letterbox_roi(h, w)))
        outs.append(est.estimate(np.zeros((h, w, 3), np.uint8))[0])
    np.testing.assert_allclose(outs[1], outs[0] * 2, rtol=1e-4)


def test_visibility_is_a_probability():
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[:33, 3] = np.linspace(-23, 9, 33)  # raw logits as emitted by the model
    est = make_estimator(raw)
    _, visibility, _ = est.estimate(np.zeros((480, 640, 3), np.uint8))
    assert visibility.min() >= 0.0 and visibility.max() <= 1.0
    assert visibility[0] < 1e-6
    assert visibility[-1] > 0.999
    assert np.all(np.diff(visibility) >= 0)


class TestPreprocess:
    def test_shape_dtype_and_range(self):
        est = PoseEstimator(model_path="unused.onnx")
        frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        out = est.preprocess(frame)
        assert out.shape == (1, 256, 256, 3)  # NHWC
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_letterbox_pads_the_short_side_with_black(self):
        est = PoseEstimator(model_path="unused.onnx")
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # wide white frame
        out = est.preprocess(frame)[0]
        assert out[0:20].max() == 0.0 and out[-20:].max() == 0.0  # top/bottom bars
        assert out[128, 128:130].min() > 0.99  # centre is image content

    def test_marker_lands_where_postprocess_expects_it(self):
        est = PoseEstimator(model_path="unused.onnx")
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[100:110, 500:510] = (0, 0, 255)  # red marker (BGR) around (505, 105)
        out = est.preprocess(frame)[0]
        ys, xs = np.nonzero(out[..., 0] > 0.5)  # channel 0 is R after BGR->RGB
        k = 256 / 640
        assert xs.mean() == pytest.approx(505 * k, abs=1.5)
        assert ys.mean() == pytest.approx(105 * k + (256 - 480 * k) / 2, abs=1.5)

    def test_roi_zooms_in(self):
        est = PoseEstimator(model_path="unused.onnx")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:280, 280:360] = 255  # 80x80 white block at the centre
        whole = est.preprocess(frame)[0].mean()
        zoomed = est.preprocess(frame, roi=(320.0, 240.0, 160.0))[0].mean()
        assert zoomed > whole * 3
