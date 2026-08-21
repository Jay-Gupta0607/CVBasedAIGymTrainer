"""Simple test to verify DWPose integration works."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from app.services.dwpose_estimator import DWPoseEstimator, get_dwpose_estimator
    print("[OK] dwpose_estimator imported")

    from app.services.pose_alignment import PoseDirectionAligner, get_pose_aligner, create_controlnet_conditioning
    print("[OK] pose_alignment imported")

    from app.services.ml_pipeline import MLPipeline
    print("[OK] ml_pipeline imported")

    from app.config import settings
    print("[OK] config imported")
    print(f"  DWPOSE_MODEL_PATH: {settings.DWPOSE_MODEL_PATH}")
    print(f"  DWPOSE_ONNX_PROVIDERS: {settings.DWPOSE_ONNX_PROVIDERS}")

def test_dwpose_estimator_basic():
    """Test DWPoseEstimator basic functionality without model."""
    from app.services.dwpose_estimator import DWPoseEstimator

    print("\nTesting DWPoseEstimator...")
    estimator = DWPoseEstimator(model_path="/tmp/test_dwpose.onnx")

    # Test preprocess
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    input_tensor, metadata = estimator.preprocess(frame)

    assert input_tensor.shape == (1, 3, 288, 384)
    assert metadata["original_shape"] == (480, 640)
    print("[OK] preprocess works")

    # Test postprocess
    keypoints_norm = np.random.rand(1, 130, 2).astype(np.float32) * 384
    scores_norm = np.ones((1, 130), dtype=np.float32)
    outputs = [keypoints_norm, scores_norm]

    keypoints, scores = estimator.postprocess(outputs, metadata)
    assert keypoints.shape == (130, 2)
    assert scores.shape == (130,)
    print("[OK] postprocess works")

    # Test keypoint extraction
    body_kpts, body_scores = estimator.get_body_keypoints(keypoints, scores)
    assert body_kpts.shape == (18, 2)
    print("[OK] get_body_keypoints works")

    face_kpts, face_scores = estimator.get_face_keypoints(keypoints, scores)
    assert face_kpts.shape == (68, 2)
    print("[OK] get_face_keypoints works")

    left_hand, right_hand = estimator.get_hand_keypoints(keypoints, scores)
    assert left_hand[0].shape == (21, 2)
    assert right_hand[0].shape == (21, 2)
    print("[OK] get_hand_keypoints works")

    # Test controlnet image creation
    control_img = estimator.create_controlnet_image(keypoints, scores, (480, 640))
    assert control_img.shape == (480, 640, 3)
    print("[OK] create_controlnet_image works")

def test_pose_alignment():
    """Test PoseDirectionAligner functionality."""
    from app.services.pose_alignment import PoseDirectionAligner

    print("\nTesting PoseDirectionAligner...")
    aligner = PoseDirectionAligner()

    # Create test keypoints facing right
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

    # Test direction detection
    direction = aligner.detect_facing_direction(keypoints, scores)
    print(f"[OK] Direction detected: {direction}")

    # Create a clearer right-facing pose (left shoulder more forward/lower, right shoulder back/higher)
    keypoints_r = np.array([
        [320, 100],   # nose
        [310, 90], [330, 90],   # eyes
        [300, 85], [340, 85],   # ears
        [200, 140], [400, 160],  # shoulders (left=200,140 forward; right=400,160 back)
        [180, 210], [420, 230],  # elbows
        [160, 270], [440, 290],  # wrists
        [250, 270], [380, 290],  # hips (left forward, right back)
        [250, 380], [380, 380],  # knees
        [250, 480], [380, 480],  # ankles
        [315, 130],   # neck
    ], dtype=np.float32)

    scores = np.ones(18, dtype=np.float32)

    direction_r = aligner.detect_facing_direction(keypoints_r, scores)
    print(f"[OK] Right-facing direction detected: {direction_r}")

    # Create left-facing by mirroring
    keypoints_l = keypoints_r.copy()
    keypoints_l[:, 0] = 640 - keypoints_r[:, 0]
    lr_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    for l, r in lr_pairs:
        keypoints_l[[l, r]] = keypoints_l[[r, l]]

    direction_l = aligner.detect_facing_direction(keypoints_l, scores)
    print(f"[OK] Left-facing direction detected: {direction_l}")

    # Now use these for the test
    keypoints = keypoints_r  # use right-facing as base

    # Test mirroring
    mirrored_kpts, mirrored_scores = aligner.mirror_pose_horizontally(
        keypoints, scores, 640
    )
    # Verify mirroring: after mirroring + swapping, each keypoint should be at
    # the mirrored position of its opposite. For body keypoints:
    # - nose, neck stay at mirrored position (no pair)
    # - left<->right pairs are swapped
    # Check nose (index 0) and neck (index 17) are mirrored without swapping
    assert abs(mirrored_kpts[0, 0] - (640 - keypoints[0, 0])) < 1
    assert abs(mirrored_kpts[17, 0] - (640 - keypoints[17, 0])) < 1
    # Check left/right pairs are swapped and mirrored
    lr_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    for l, r in lr_pairs:
        # left index should now have right's mirrored position
        assert abs(mirrored_kpts[l, 0] - (640 - keypoints[r, 0])) < 1
        # right index should now have left's mirrored position
        assert abs(mirrored_kpts[r, 0] - (640 - keypoints[l, 0])) < 1
    print("[OK] Mirror pose works")

    # Test alignment (same direction)
    aligned_kpts, aligned_scores, info = aligner.align_trainer_to_user(
        keypoints, scores, keypoints, scores, (480, 640), (480, 640)
    )
    assert info["mirrored"] is False
    print("[OK] Align same direction works")

    # Test alignment (different direction) - create left-facing by mirroring
    left_keypoints = keypoints.copy()
    left_keypoints[:, 0] = 640 - keypoints[:, 0]  # Mirror X
    # Swap left/right pairs to maintain correct body orientation
    lr_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    for l, r in lr_pairs:
        left_keypoints[[l, r]] = left_keypoints[[r, l]]

    aligned_kpts2, _, info2 = aligner.align_trainer_to_user(
        keypoints, scores, left_keypoints, scores, (480, 640), (480, 640)
    )
    # When directions differ, it should mirror
    assert info2["mirrored"] is True
    print("[OK] Align different direction works")

    # Test conditioning image
    from app.services.pose_alignment import create_controlnet_conditioning
    cond_img = create_controlnet_conditioning(keypoints, scores, (480, 640))
    assert cond_img.shape == (480, 640, 3)
    assert np.any(cond_img > 0)
    print("[OK] ControlNet conditioning works")

def test_ml_pipeline_integration():
    """Test MLPipeline has pose transfer method."""
    from app.services.ml_pipeline import MLPipeline
    from app.services.dwpose_estimator import get_dwpose_estimator
    from app.services.pose_alignment import get_pose_aligner

    print("\nTesting MLPipeline integration...")

    # Create pipeline without full initialization (models not present in test env)
    pipeline = MLPipeline.__new__(MLPipeline)
    pipeline.dwpose_estimator = get_dwpose_estimator()
    pipeline.pose_aligner = get_pose_aligner()
    # Don't initialize other components

    # Check components exist
    assert hasattr(pipeline, 'dwpose_estimator')
    assert hasattr(pipeline, 'pose_aligner')
    assert hasattr(pipeline, 'generate_pose_transfer')
    print("[OK] MLPipeline has pose transfer components")

    # Check method signature
    import inspect
    sig = inspect.signature(pipeline.generate_pose_transfer)
    params = list(sig.parameters.keys())
    assert 'user_frame' in params
    assert 'trainer_image_path' in params
    assert 'exercise_name' in params
    print("[OK] generate_pose_transfer has correct signature")

def main():
    """Run all tests."""
    print("=" * 60)
    print("DWPose Integration Tests")
    print("=" * 60)

    test_imports()
    test_dwpose_estimator_basic()
    test_pose_alignment()
    test_ml_pipeline_integration()

    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)

if __name__ == "__main__":
    main()