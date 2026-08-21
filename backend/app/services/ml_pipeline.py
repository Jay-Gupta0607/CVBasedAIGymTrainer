"""Main ML pipeline orchestrator."""

import base64
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

import cv2
import httpx
import numpy as np

from app.config import settings
from app.services.pose_estimator import PoseEstimator, get_pose_estimator
from app.services.dwpose_estimator import DWPoseEstimator, get_dwpose_estimator
from app.services.pose_alignment import PoseDirectionAligner, get_pose_aligner, create_controlnet_conditioning
from app.services.dtw_aligner import (
    DTWAligner,
    compute_joint_angles,
    compute_angle_differences,
)
from app.services.form_scorer import FormScorer, RuleBasedScorer, create_scorer
from app.services.video_processor import VideoProcessor, get_video_processor
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)


class MLPipeline:
    """End-to-end ML pipeline for exercise form analysis."""

    def __init__(
        self,
        pose_estimator: Optional[PoseEstimator] = None,
        dwpose_estimator: Optional[DWPoseEstimator] = None,
        pose_aligner: Optional[PoseDirectionAligner] = None,
        dtw_aligner: Optional[DTWAligner] = None,
        form_scorer: Optional[FormScorer] = None,
        rule_scorer: Optional[RuleBasedScorer] = None,
        video_processor: Optional[VideoProcessor] = None,
    ):
        self.pose_estimator = pose_estimator or get_pose_estimator()
        self.dwpose_estimator = dwpose_estimator or get_dwpose_estimator()
        self.pose_aligner = pose_aligner or get_pose_aligner()
        self.dtw_aligner = dtw_aligner or DTWAligner(self.pose_estimator)
        self.form_scorer = form_scorer or create_scorer()
        self.rule_scorer = rule_scorer or RuleBasedScorer()
        self.video_processor = video_processor or get_video_processor()
        self.storage = get_storage_service()

        # Initialize all components
        self.pose_estimator.initialize()
        self.dwpose_estimator.initialize()
        self.form_scorer.initialize()

    def analyze_with_trainer(
        self,
        user_video_path: str,
        trainer_video_path: str,
        exercise_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Analyze user video against trainer reference.

        Args:
            user_video_path: Path to user video
            trainer_video_path: Path to trainer video
            exercise_name: Exercise type
            progress_callback: Optional callback(progress: int, stage: str)

        Returns:
            Analysis results dict
        """
        start_time = time.time()

        try:
            # Stage 1: Extract frames
            if progress_callback:
                progress_callback(10, "Extracting frames from videos")

            user_frames, user_meta = self.video_processor.extract_frames(user_video_path)
            trainer_frames, trainer_meta = self.video_processor.extract_frames(trainer_video_path)

            logger.info(f"User: {len(user_frames)} frames, Trainer: {len(trainer_frames)} frames")

            # Stage 2: Pose estimation
            if progress_callback:
                progress_callback(25, "Estimating poses")

            user_poses = self._estimate_poses_batch(user_frames, "user")
            trainer_poses = self._estimate_poses_batch(trainer_frames, "trainer")

            # Stage 3: DTW alignment
            if progress_callback:
                progress_callback(50, "Aligning sequences")

            user_embeddings = np.array([p["embedding"] for p in user_poses])
            trainer_embeddings = np.array([p["embedding"] for p in trainer_poses])

            warping_path, dtw_distance = self.dtw_aligner.align_sequences(
                user_embeddings, trainer_embeddings
            )

            # Stage 4: Frame-by-frame analysis
            if progress_callback:
                progress_callback(70, "Analyzing form frame by frame")

            frame_results = []
            total_score = 0
            violations_per_rep: Dict[int, List] = {}

            # Get repetition segments
            user_visibilities = np.array([p["visibility"] for p in user_poses])
            segments = self.dtw_aligner.segment_repetitions(
                user_embeddings, user_visibilities, exercise_name
            )

            for i, (user_emb_idx, trainer_emb_idx) in enumerate(warping_path):
                if progress_callback and i % max(1, len(warping_path) // 20) == 0:
                    progress = 70 + int(20 * i / len(warping_path))
                    progress_callback(progress, f"Analyzing frame {i+1}/{len(warping_path)}")

                if (user_emb_idx >= len(user_poses) or
                    trainer_emb_idx >= len(trainer_poses)):
                    continue

                user_pose = user_poses[user_emb_idx]
                trainer_pose = trainer_poses[trainer_emb_idx]

                # Compute joint angles
                user_angles = compute_joint_angles(user_pose["landmarks"])
                trainer_angles = compute_joint_angles(trainer_pose["landmarks"])

                # Score frame
                error_score, contributions = self.form_scorer.score_frame(
                    user_pose["landmarks"],
                    user_pose["visibility"],
                    trainer_pose["landmarks"],
                    trainer_pose["visibility"],
                    user_angles,
                    trainer_angles,
                )

                # Rule-based analysis for detailed feedback
                rule_score, violations = self.rule_scorer.score_frame(
                    exercise_name, user_angles, trainer_angles
                )

                # Combine scores (use ML score as primary, rule as validation)
                final_score = error_score

                # Encode images as base64
                user_frame = user_frames[user_pose["frame_idx"]]
                trainer_frame = trainer_frames[trainer_pose["frame_idx"]]

                user_image_b64 = self.video_processor.save_frame_as_base64(user_frame)
                trainer_image_b64 = self.video_processor.save_frame_as_base64(trainer_frame)

                frame_result = {
                    "frame_id": user_pose["frame_idx"],
                    "error_score": final_score,
                    "feedback": self._generate_feedback(violations, user_angles, trainer_angles),
                    "technical_observation": self._generate_technical_obs(
                        user_angles, trainer_angles, violations
                    ),
                    "user_image": user_image_b64,
                    "trainer_image": trainer_image_b64,
                    "joint_angles": user_angles,
                    "angle_differences": compute_angle_differences(user_angles, trainer_angles),
                }
                frame_results.append(frame_result)
                total_score += final_score

                # Track violations per rep
                rep_idx = self._find_rep_index(user_pose["frame_idx"], segments)
                if rep_idx not in violations_per_rep:
                    violations_per_rep[rep_idx] = []
                violations_per_rep[rep_idx].extend(violations)

            # Stage 5: Aggregate results
            if progress_callback:
                progress_callback(95, "Generating summary")

            avg_score = total_score / len(frame_results) if frame_results else 0
            reps = len(segments)

            feedback_summary = self._generate_summary(exercise_name, avg_score, reps, violations_per_rep)
            technical_details = self._generate_technical_details(violations_per_rep)

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(100, "Complete")

            return {
                "analysis": frame_results,
                "reps": reps,
                "feedback_summary": feedback_summary,
                "technical_details": technical_details,
                "processing_time": processing_time,
                "dtw_distance": dtw_distance,
                "segments": segments,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    def analyze_without_trainer(
        self,
        user_video_path: str,
        exercise_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Analyze user video without trainer reference (self-consistency check)."""
        start_time = time.time()

        try:
            if progress_callback:
                progress_callback(10, "Extracting frames")

            user_frames, user_meta = self.video_processor.extract_frames(user_video_path)

            if progress_callback:
                progress_callback(25, "Estimating poses")

            user_poses = self._estimate_poses_batch(user_frames, "user")

            if progress_callback:
                progress_callback(50, "Analyzing form")

            # Analyze each frame against ideal form rules
            frame_results = []
            total_score = 0

            user_visibilities = np.array([p["visibility"] for p in user_poses])
            segments = self.dtw_aligner.segment_repetitions(
                np.array([p["embedding"] for p in user_poses]),
                user_visibilities,
                exercise_name
            )

            for i, user_pose in enumerate(user_poses):
                if progress_callback and i % max(1, len(user_poses) // 20) == 0:
                    progress = 50 + int(40 * i / len(user_poses))
                    progress_callback(progress, f"Analyzing frame {i+1}/{len(user_poses)}")

                user_angles = compute_joint_angles(user_pose["landmarks"])

                # Score using rule-based only (no trainer comparison)
                rule_score, violations = self.rule_scorer.score_frame(
                    exercise_name, user_angles, {}
                )

                user_frame = user_frames[user_pose["frame_idx"]]
                user_image_b64 = self.video_processor.save_frame_as_base64(user_frame)

                frame_result = {
                    "frame_id": user_pose["frame_idx"],
                    "error_score": rule_score,
                    "feedback": self._generate_feedback(violations, user_angles, {}),
                    "technical_observation": self._generate_technical_obs(
                        user_angles, {}, violations
                    ),
                    "user_image": user_image_b64,
                    "trainer_image": "",  # No trainer
                    "joint_angles": user_angles,
                }
                frame_results.append(frame_result)
                total_score += rule_score

            avg_score = total_score / len(frame_results) if frame_results else 0
            reps = len(segments)

            feedback_summary = self._generate_summary(exercise_name, avg_score, reps, {})
            technical_details = self._generate_technical_details({})

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(100, "Complete")

            return {
                "analysis": frame_results,
                "reps": reps,
                "feedback_summary": feedback_summary,
                "technical_details": technical_details,
                "processing_time": processing_time,
                "segments": segments,
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    def _estimate_poses_batch(
        self,
        frames: List[np.ndarray],
        source: str,
    ) -> List[Dict]:
        """Estimate poses for batch of frames."""
        results = []

        for idx, frame in enumerate(frames):
            landmarks, visibility, meta = self.pose_estimator.estimate(frame)

            # Create pose embedding (flattened visible landmarks)
            embedding = np.zeros(99, dtype=np.float32)
            for i in range(33):
                if visibility[i] > 0.5:
                    embedding[i*3:(i+1)*3] = landmarks[i]

            results.append({
                "frame_idx": idx,
                "landmarks": landmarks,
                "visibility": visibility,
                "embedding": embedding,
                "meta": meta,
            })

        return results

    def _find_rep_index(self, frame_idx: int, segments: List[Tuple[int, int]]) -> int:
        """Find which rep segment a frame belongs to."""
        for i, (start, end) in enumerate(segments):
            if start <= frame_idx <= end:
                return i
        return -1

    def _generate_feedback(
        self,
        violations: List[Dict],
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
    ) -> str:
        """Generate human-readable feedback from violations."""
        if not violations:
            return "Good form! Keep it up."

        feedback_parts = []
        for v in violations[:3]:  # Top 3 issues
            joint = v["joint"].replace("_", " ").title()
            issue = v["issue"]
            feedback_parts.append(f"{joint}: {issue}")

        return "; ".join(feedback_parts) + "."

    def _generate_technical_obs(
        self,
        user_angles: Dict[str, float],
        trainer_angles: Dict[str, float],
        violations: List[Dict],
    ) -> str:
        """Generate technical observation."""
        lines = []

        # Show key angle comparisons
        key_angles = ["left_elbow", "right_elbow", "left_knee", "right_knee",
                      "left_hip", "right_hip", "left_shoulder", "right_shoulder", "torso_lean"]

        for angle in key_angles:
            user_val = user_angles.get(angle)
            trainer_val = trainer_angles.get(angle)
            if user_val is not None:
                diff = abs(user_val - (trainer_val or user_val))
                angle_name = angle.replace("_", " ").title()
                target = f" (target: {trainer_val:.1f}°)" if trainer_val else ""
                lines.append(f"{angle_name}: {user_val:.1f}°{target} [Δ{diff:.1f}°]")

        if violations:
            lines.append("\nCorrections needed:")
            for v in violations[:3]:
                lines.append(f"  - {v['joint'].replace('_', ' ').title()}: {v['issue']}")

        return "\n".join(lines)

    def _generate_summary(
        self,
        exercise_name: str,
        avg_score: float,
        reps: int,
        violations_per_rep: Dict[int, List],
    ) -> str:
        """Generate overall feedback summary."""
        score_desc = "excellent" if avg_score < 20 else \
                     "good" if avg_score < 40 else \
                     "needs improvement" if avg_score < 60 else \
                     "poor"

        total_violations = sum(len(v) for v in violations_per_rep.values())

        return (
            f"Completed {reps} repetition(s) of {exercise_name}. "
            f"Overall form score: {score_desc} ({100 - avg_score}/100). "
            f"Detected {total_violations} form issue(s) across repetitions. "
            f"Focus on the corrections below for improvement."
        )

    def _generate_technical_details(
        self,
        violations_per_rep: Dict[int, List],
    ) -> List[Dict]:
        """Generate detailed technical corrections."""
        details = []

        # Aggregate violations by joint
        joint_issues = {}
        for rep_violations in violations_per_rep.values():
            for v in rep_violations:
                joint = v["joint"]
                if joint not in joint_issues:
                    joint_issues[joint] = []
                joint_issues[joint].append(v)

        for joint, issues in joint_issues.items():
            # Get most common issue
            from collections import Counter
            issue_counts = Counter(v["issue"] for v in issues)
            main_issue = issue_counts.most_common(1)[0][0]
            avg_severity = np.mean([v["severity"] for v in issues])

            details.append({
                "title": joint.replace("_", " ").title(),
                "description": f"{main_issue} (occurred in {len(issues)} rep(s), "
                              f"avg severity: {avg_severity:.1f}/1.0)",
            })

        return details

    def generate_pose_transfer(
        self,
        user_frame: np.ndarray,
        trainer_image_path: str,
        exercise_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Generate pose-transfer image for a single frame.

        Args:
            user_frame: User video frame (BGR)
            trainer_image_path: Path to trainer reference image
            exercise_name: Exercise type for prompting
            progress_callback: Optional callback(progress: int, stage: str)

        Returns:
            Dict with "image_base64", "pose_alignment", "control_image_base64"
        """
        start_time = time.time()

        try:
            # Stage 1: Load trainer reference image
            if progress_callback:
                progress_callback(10, "Loading trainer reference")

            trainer_frame = cv2.imread(trainer_image_path)
            if trainer_frame is None:
                raise ValueError(f"Could not load trainer image: {trainer_image_path}")

            # Stage 2: Extract DWPose from both images
            if progress_callback:
                progress_callback(30, "Extracting poses (DWPose)")

            user_kpts, user_scores, user_meta = self.dwpose_estimator.estimate(user_frame)
            trainer_kpts, trainer_scores, trainer_meta = self.dwpose_estimator.estimate(trainer_frame)

            # Stage 3: Align directions
            if progress_callback:
                progress_callback(50, "Aligning pose directions")

            aligned_trainer_kpts, aligned_trainer_scores, alignment_info = \
                self.pose_aligner.align_trainer_to_user(
                    user_kpts, user_scores,
                    trainer_kpts, trainer_scores,
                    user_frame.shape, trainer_frame.shape,
                    full_keypoints=True,
                )

            # Stage 4: Create ControlNet conditioning image
            if progress_callback:
                progress_callback(70, "Creating ControlNet conditioning")

            control_image = create_controlnet_conditioning(
                aligned_trainer_kpts,
                aligned_trainer_scores,
                user_frame.shape[:2],
                min_score=0.3,
            )

            # Encode images as base64
            _, user_buf = cv2.imencode('.jpg', user_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, trainer_buf = cv2.imencode('.jpg', trainer_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, control_buf = cv2.imencode('.jpg', control_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            user_b64 = base64.b64encode(user_buf).decode('utf-8')
            trainer_b64 = base64.b64encode(trainer_buf).decode('utf-8')
            control_b64 = base64.b64encode(control_buf).decode('utf-8')

            # Stage 5: Call Modal endpoint for generation
            if progress_callback:
                progress_callback(80, "Generating pose transfer image")

            generated_image_b64 = self._call_modal_pose_transfer(
                user_image_b64=user_b64,
                trainer_image_b64=trainer_b64,
                control_image_b64=control_b64,
                exercise_name=exercise_name,
            )

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(100, "Complete")

            return {
                "image_base64": generated_image_b64,
                "control_image_base64": control_b64,
                "pose_alignment": alignment_info,
                "processing_time": processing_time,
                "user_keypoints": user_kpts.tolist(),
                "trainer_keypoints": trainer_kpts.tolist(),
                "aligned_trainer_keypoints": aligned_trainer_kpts.tolist(),
            }

        except Exception as e:
            logger.error(f"Pose transfer generation failed: {e}", exc_info=True)
            raise

    def _call_modal_pose_transfer(
        self,
        user_image_b64: str,
        trainer_image_b64: str,
        control_image_b64: str,
        exercise_name: str,
    ) -> str:
        """Call Modal endpoint for pose transfer generation.

        Returns:
            Base64 encoded generated image
        """
        modal_endpoint = settings.MODAL_POSE_TRANSFER_ENDPOINT
        if not modal_endpoint:
            raise ValueError("MODAL_POSE_TRANSFER_ENDPOINT not configured")

        # Prepare multipart form data
        import io
        user_bytes = base64.b64decode(user_image_b64)
        trainer_bytes = base64.b64decode(trainer_image_b64)
        control_bytes = base64.b64decode(control_image_b64)

        # Exercise-specific prompts for better results
        exercise_prompts = {
            "Squat": "professional squat form, perfect depth, knees tracking toes, upright torso, gym lighting, high quality fitness photography",
            "Deadlift": "professional deadlift form, neutral spine, hip hinge, bar path vertical, gym lighting, high quality fitness photography",
            "Bench Press": "professional bench press form, retracted scapulae, elbows 45 degrees, full range of motion, gym lighting",
            "Push-up": "perfect push-up form, straight body line, elbows 45 degrees, chest to floor, gym lighting",
            "Pull Up": "strict pull-up form, chin over bar, full extension, engaged lats, gym lighting",
            "Shoulder Press": "strict overhead press, vertical bar path, locked out elbows, core engaged, gym lighting",
        }

        prompt = exercise_prompts.get(
            exercise_name,
            f"professional {exercise_name.lower()} form, perfect technique, gym lighting, high quality"
        )

        files = {
            "user_image": ("user.jpg", io.BytesIO(user_bytes), "image/jpeg"),
            "trainer_image": ("trainer.jpg", io.BytesIO(trainer_bytes), "image/jpeg"),
            "control_image": ("control.jpg", io.BytesIO(control_bytes), "image/jpeg"),
        }
        data = {"prompt": prompt}

        # Use synchronous httpx for now (can be made async if needed)
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"{modal_endpoint}/pose-transfer",
                files=files,
                data=data,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Modal API error: {response.status_code} - {response.text}")

            # Encode response as base64
            return base64.b64encode(response.content).decode('utf-8')


async def run_full_analysis(
    user_video_path: str,
    trainer_video_path: Optional[str],
    exercise_name: str,
    email: str,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Convenience function to run full analysis."""
    pipeline = MLPipeline()

    if trainer_video_path:
        return pipeline.analyze_with_trainer(
            user_video_path, trainer_video_path, exercise_name, progress_callback
        )
    else:
        return pipeline.analyze_without_trainer(
            user_video_path, exercise_name, progress_callback
        )