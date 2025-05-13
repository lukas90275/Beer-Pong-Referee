import cv2
import numpy as np

from trackers.object_tracker import ObjectTracker, TrackerConfig


class TableTracker(ObjectTracker):
    """
    Specific implementation for tracking a table.
    Tables are typically stationary in the world frame, so velocity is
    mostly due to camera motion which is compensated at the manager level.
    """

    def __init__(self, initial_box, frame_shape, **kwargs):
        config = TrackerConfig(
            tracker_type="Table",
            color=(0, 255, 0),
            max_lost_frames=5,
            min_confidence_frames=3,
            search_expansion_factor=1.3,
            velocity_smoothing=0.9,
            velocity_threshold=2.0
        )
        super().__init__(
            initial_box,
            frame_shape,
            config=config,
            **kwargs
        ) 