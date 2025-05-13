from typing import Dict, Optional, Tuple

import cv2

from frame_analysis import analyze_frame
from video_process.video_processor import VideoProcessor


class VideoPreprocessor:
    def __init__(self, video_path: str, padding: Optional[Dict[str, float]] = None):
        """
        Initialize the video preprocessor.

        Args:
            video_path (str): Path to the video file
            padding (Dict[str, float], optional): Padding ratios for bounds.
                Defaults to {'left': 0.3, 'right': 0.3, 'top': 4, 'bottom': 0.05}
        """
        self.video_path = video_path
        self.padding = padding or {
            "left": 0.4,
            "right": 0.4,
            "top": 4,
            "bottom": 0.1,
        }

        self.processor = VideoProcessor(video_path)

        self.crop_bounds = None

    def _calculate_padded_bounds(self, bounds: Dict[str, int]) -> Dict[str, int]:
        """Calculate padded bounds based on table detection."""
        table_width = bounds["x2"] - bounds["x1"]
        table_height = bounds["y2"] - bounds["y1"]

        return {
            "x1": max(0, int(bounds["x1"] - table_width * self.padding["left"])),
            "x2": min(
                int(self.processor.video_info["width"]),
                int(bounds["x2"] + table_width * self.padding["right"]),
            ),
            "y1": max(0, int(bounds["y1"] - table_height * self.padding["top"])),
            "y2": min(
                int(self.processor.video_info["height"]),
                int(bounds["y2"] + table_height * self.padding["bottom"]),
            ),
        }

    def detect_stable_bounds(self, max_frames: int = 30) -> Dict[str, int]:
        """
        Analyze frames until a stable table tracker is established.
        Once a confident tracker is found, use its bounds immediately.

        Args:
            max_frames (int): Maximum number of frames to analyze

        Returns:
            Dict[str, int]: Stable cropping bounds
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        last_table_state = None

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            _, detections = analyze_frame(frame)

            table_tracker_state = detections.get("table_tracker")

            if table_tracker_state:
                last_table_state = table_tracker_state

            if table_tracker_state and table_tracker_state["is_confident"]:
                box = table_tracker_state["box"]
                table_bounds = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}

                self.crop_bounds = self._calculate_padded_bounds(table_bounds)
                cap.release()
                return self.crop_bounds

            frame_count += 1

        cap.release()

        if last_table_state:
            box = last_table_state["box"]
            table_bounds = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
            self.crop_bounds = self._calculate_padded_bounds(table_bounds)
            return self.crop_bounds

        raise ValueError(
            "No table tracker state found within max_frames. Check analyze_frame logic or video segment."
        )

    def process_video(self, output_path: str) -> None:
        """
        Process the entire video with the detected bounds and save to output_path.

        Args:
            output_path (str): Path to save the processed video
        """
        if self.crop_bounds is None:
            self.detect_stable_bounds()

        input_cap = cv2.VideoCapture(self.video_path)
        width = self.crop_bounds["x2"] - self.crop_bounds["x1"]
        height = self.crop_bounds["y2"] - self.crop_bounds["y1"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.processor.fps, (width, height))

        while True:
            ret, frame = input_cap.read()
            if not ret:
                break

            cropped = frame[
                self.crop_bounds["y1"] : self.crop_bounds["y2"],
                self.crop_bounds["x1"] : self.crop_bounds["x2"],
            ]

            out.write(cropped)

        input_cap.release()
        out.release()

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the processed frames."""
        if self.crop_bounds is None:
            self.detect_stable_bounds()

        width = self.crop_bounds["x2"] - self.crop_bounds["x1"]
        height = self.crop_bounds["y2"] - self.crop_bounds["y1"]
        return width, height
