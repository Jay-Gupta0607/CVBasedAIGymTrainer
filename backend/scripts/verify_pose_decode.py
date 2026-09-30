"""Check the ONNX pose pipeline against MediaPipe's official PoseLandmarker.

Run it on real footage (e.g. a side-on squat clip) to see how closely `PoseEstimator`
agrees with MediaPipe, both on landmark positions and on the joint angles the scoring
uses. The reference is MediaPipe itself (person detector + cropped ROI), so this
measures agreement with MediaPipe, not with ground truth.

    pip install -r backend/requirements/ml.txt        # provides mediapipe
    python backend/scripts/verify_pose_decode.py squat.mp4 --every 5 --overlay out.mp4
    python backend/scripts/verify_pose_decode.py squat.mp4 --refine     # also try a tight-ROI 2nd pass

Needs models/pose_landmarker.onnx and models/pose_landmarker_full.task (see README).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pose_estimator import PoseEstimator, SKELETON_CONNECTIONS  # noqa: E402
from app.services.pose_features import compute_joint_angles, torso_length  # noqa: E402

VISIBLE_AT = 0.5
KEY_LANDMARKS = {
    "shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
    "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28),
}
ANGLES = ("left_knee", "right_knee", "left_hip", "right_hip", "left_elbow", "right_elbow", "torso_lean")


def roi_from_landmarks(landmarks, visibility, margin=1.25, min_visible=8):
    """Square ROI around the visible landmarks, or None if too few are visible."""
    visible = visibility >= VISIBLE_AT
    if visible.sum() < min_visible:
        return None
    x0, y0 = landmarks[visible, :2].min(axis=0)
    x1, y1 = landmarks[visible, :2].max(axis=0)
    return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * margin


def load_reference(task_path):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.exit("mediapipe is required: pip install -r backend/requirements/ml.txt")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    def run(frame_bgr):
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        pts = result.pose_landmarks[0]
        landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in pts], dtype=np.float32)
        visibility = np.array([p.visibility for p in pts], dtype=np.float32)
        return landmarks, visibility

    return run


def frames_from(path, every, max_frames):
    image = cv2.imread(str(path))
    if image is not None:
        yield 0, image
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        sys.exit(f"Cannot open {path}")
    index = taken = 0
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and taken >= max_frames):
            break
        if index % every == 0:
            yield index, frame
            taken += 1
        index += 1
    capture.release()


def draw(frame, landmarks, visibility, color):
    for a, b in SKELETON_CONNECTIONS:
        if visibility[a] >= VISIBLE_AT and visibility[b] >= VISIBLE_AT:
            cv2.line(frame, tuple(int(v) for v in landmarks[a, :2]), tuple(int(v) for v in landmarks[b, :2]), color, 2)


def report(title, records):
    if not records["err"]:
        print(f"\n{title}: no frames to compare")
        return
    err = np.array(records["err"])
    print(f"\n{title}  ({len(err)} frames)")
    print(f"  landmark error / torso length : mean {err.mean():.3f}  median {np.median(err):.3f}  p90 {np.percentile(err, 90):.3f}")
    print("  per landmark (mean error / torso length):  "
          + "  ".join(f"{n} {np.mean(v):.2f}" for n, v in records["per_lm"].items() if v))
    print("  joint angle |ours - official| (degrees):")
    for name in ANGLES:
        diffs = records["angle"].get(name)
        if diffs:
            d = np.array(diffs)
            print(f"    {name:14s} MAE {d.mean():5.1f}   p90 {np.percentile(d, 90):5.1f}   (n={len(d)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="image or video")
    parser.add_argument("--onnx", type=Path, default=Path("models/pose_landmarker.onnx"))
    parser.add_argument("--task", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--every", type=int, default=5, help="use every Nth video frame")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N compared frames (0 = all)")
    parser.add_argument("--refine", action="store_true", help="also evaluate a second pass on a tight ROI")
    parser.add_argument("--overlay", type=Path, help="write an image/video: green = ours, red = official")
    args = parser.parse_args()

    for path in (args.onnx, args.task):
        if not path.exists():
            sys.exit(f"Missing {path} - see 'Download the ML Models' in the README")

    estimator = PoseEstimator(model_path=str(args.onnx))
    estimator.initialize()
    reference = load_reference(args.task)

    modes = ["letterbox"] + (["refined ROI"] if args.refine else [])
    records = {m: {"err": [], "per_lm": {n: [] for n in KEY_LANDMARKS}, "angle": {}} for m in modes}
    skipped = 0
    writer = None

    for index, frame in frames_from(args.input, args.every, args.max_frames):
        ref = reference(frame)
        if ref is None:
            skipped += 1
            continue
        ref_lm, ref_vis = ref
        scale = torso_length(ref_lm)
        if scale < 1e-6:
            skipped += 1
            continue
        ref_angles = compute_joint_angles(ref_lm, ref_vis)
        visible = ref_vis >= VISIBLE_AT

        outputs = {}
        first = estimator.estimate(frame)[:2]
        outputs["letterbox"] = first
        if args.refine:
            roi = roi_from_landmarks(*first)
            outputs["refined ROI"] = estimator.estimate(frame, roi)[:2] if roi else first

        for mode, (lm, vis) in outputs.items():
            rec = records[mode]
            dist = np.linalg.norm(lm[:, :2] - ref_lm[:, :2], axis=1) / scale
            rec["err"].append(float(dist[visible].mean()))
            for name, idxs in KEY_LANDMARKS.items():
                sel = [i for i in idxs if visible[i]]
                if sel:
                    rec["per_lm"][name].append(float(dist[sel].mean()))
            for name, value in compute_joint_angles(lm, vis).items():
                if name in ref_angles and name in ANGLES:
                    rec["angle"].setdefault(name, []).append(abs(value - ref_angles[name]))

        if args.overlay:
            canvas = frame.copy()
            draw(canvas, ref_lm, ref_vis, (0, 0, 255))
            draw(canvas, *outputs[modes[-1]], (0, 200, 0))
            if args.input.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                cv2.imwrite(str(args.overlay), canvas)
            else:
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(str(args.overlay), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(canvas)

    if writer is not None:
        writer.release()

    print(f"\nInput: {args.input}   reference frames without a detected pose: {skipped}")
    for mode in modes:
        report(f"[{mode}]", records[mode])
    if args.overlay:
        print(f"\nOverlay written to {args.overlay}  (green = ours, red = MediaPipe official)")


if __name__ == "__main__":
    main()
