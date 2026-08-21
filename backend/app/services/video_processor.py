"""Video processing utilities."""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing utilities for frame extraction, encoding, etc."""

    def __init__(
        self,
        target_fps: int = 30,
        max_frames: int = 900,  # 30 seconds at 30fps
    ):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        target_fps: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], dict]:
        """Extract frames from video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
            target_fps: Target FPS for sampling

        Returns:
            Tuple of (frames_list, metadata_dict)
        """
        max_frames = max_frames or self.max_frames
        target_fps = target_fps or self.target_fps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # Calculate sampling interval
        if target_fps >= original_fps:
            sample_every = 1
        else:
            sample_every = max(1, int(original_fps / target_fps))

        frames = []
        frame_indices = []
        frame_count = 0
        extracted = 0

        while extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_every == 0:
                frames.append(frame)
                frame_indices.append(frame_count)
                extracted += 1

            frame_count += 1

        cap.release()

        metadata = {
            "original_fps": original_fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration,
            "extracted_frames": len(frames),
            "frame_indices": frame_indices,
            "sample_every": sample_every,
        }

        logger.info(f"Extracted {len(frames)} frames from {video_path} "
                   f"({width}x{height}, {original_fps:.1f}fps, {duration:.1f}s)")

        return frames, metadata

    def extract_frames_generator(
        self,
        video_path: str,
        target_fps: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        """Generator that yields frames one at a time (memory efficient).

        Yields:
            Tuple of (frame, frame_index, metadata)
        """
        target_fps = target_fps or self.target_fps

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if target_fps >= original_fps:
            sample_every = 1
        else:
            sample_every = max(1, int(original_fps / target_fps))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_every == 0:
                metadata = {
                    "frame_count": frame_count,
                    "original_fps": original_fps,
                    "width": width,
                    "height": height,
                }
                yield frame, frame_count, metadata

            frame_count += 1

        cap.release()

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata without extracting frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

        cap.release()
        return info

    def save_frame_as_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode frame as base64 JPEG string."""
        import base64

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

    def save_frames_as_base64(
        self,
        frames: List[np.ndarray],
        quality: int = 85,
    ) -> List[str]:
        """Encode multiple frames as base64."""
        return [self.save_frame_as_base64(f, quality) for f in frames]

    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
        codec: str = "mp4v",
    ) -> None:
        """Create video file from frames."""
        if not frames:
            raise ValueError("No frames provided")

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        logger.info(f"Created video: {output_path} ({len(frames)} frames, {fps}fps)")

    def validate_video(self, video_path: str, max_size_mb: int = 500) -> Tuple[bool, str]:
        """Validate video file."""
        path = Path(video_path)

        if not path.exists():
            return False, "File does not exist"

        # Check size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File size {size_mb:.1f}MB exceeds limit of {max_size_mb}MB"

        # Check if readable
        try:
            info = self.get_video_info(video_path)
            if info["total_frames"] == 0:
                return False, "Video has no frames"
            if info["duration"] <= 0:
                return False, "Invalid video duration"
        except Exception as e:
            return False, f"Cannot read video: {e}"

        return True, "OK"


def get_video_processor() -> VideoProcessor:
    """Get video processor instance."""
    return VideoProcessor()