import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class CupTracker(ObjectTracker):
    """
    Specific implementation for tracking cups on the beer pong table.
    Cups are mostly stationary and should only move when hit or manipulated.
    """

    def __init__(
        self, 
        initial_box, 
        frame_shape, 
        search_region_bounds=None,
        **kwargs
    ):
        config = TrackerConfig(
            tracker_type="Cup",
            color=(0, 0, 255), 
            search_expansion_factor=1.2, 
            velocity_smoothing=0.95, 
            velocity_threshold=3.0,
            position_stability_factor=0.95, 
            use_x_distance_only=True, 
            search_region_bounds=search_region_bounds,
            detect_aspect_ratio_change=True,
            aspect_ratio_change_factor=1.5,
            aspect_ratio_history_len=5
        )
        super().__init__(
            initial_box,
            frame_shape,
            config=config,
            **kwargs
        ) 