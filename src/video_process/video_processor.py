from collections.abc import Iterator
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, video_path: str):
        """
        Initialize the video processor with a video file path.

        Args:
            video_path (str): Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

    def timestamp_to_frame(self, timestamp: float) -> int:
        """
        Convert a timestamp (in seconds) to frame number.

        Args:
            timestamp (float): Time in seconds

        Returns:
            int: Corresponding frame number
        """
        return int(timestamp * self.fps)

    def process_video_segment(
        self,
        start_time: float,
        end_time: float,
        frame_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        skip_frames: int = 0,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Process a segment of the video between start_time and end_time.

        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            frame_processor (Callable): Function to process each frame and return the processed frame
            skip_frames (int): Number of frames to skip between processed frames

        Yields:
            Tuple[int, np.ndarray]: Tuple of (frame_number, processed_frame)
        """
        start_frame = self.timestamp_to_frame(start_time)
        end_frame = self.timestamp_to_frame(end_time)

        if start_time < 0 or end_time > self.duration or start_time >= end_time:
            raise ValueError("Invalid time range specified")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        frames_processed = 0

        while current_frame <= end_frame:
            ret, frame = self.cap.read()

            if not ret:
                break

            if frames_processed % (skip_frames + 1) == 0:
                if frame_processor is not None:
                    processed_frame = frame_processor(frame)
                    yield current_frame, processed_frame
                else:
                    yield current_frame, frame

            current_frame += 1
            frames_processed += 1

    def __del__(self):
        """Release video capture resources"""
        if hasattr(self, "cap"):
            self.cap.release()

    @property
    def video_info(self) -> dict:
        """Return basic information about the video"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
