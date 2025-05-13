from trackers.table_tracker import TableTracker
from tracking_managers.tracker_manager import TrackerManager


class TableTrackerManager(TrackerManager):
    """
    Manages TableTracker instances.
    Simple implementation that tracks a single table instance.
    """

    def __init__(
        self,
        iou_threshold=0.2,
        min_confidence_frames=3,
        max_lost_frames=5,
        detection_threshold=0.7,
        **kwargs
    ):
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=True,
            motion_history_max_len=8,
            target_labels=["dining table", "table"],
            **kwargs
        )
        
        self.register_tracker_factory(self._create_table_tracker)

    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new tracker from a detection - implements abstract method"""
        return self._create_table_tracker(detection, frame_shape)

    def _create_table_tracker(self, detection, frame_shape):
        """Create a new TableTracker from a detection"""
        return TableTracker(
            initial_box=self._extract_bbox_from_detection(detection),
            frame_shape=frame_shape,
            initial_confidence=detection["confidence"],
            min_confidence_frames=self.min_confidence_frames,
            max_lost_frames=self.max_lost_frames,
        )

    def get_primary_tracker(self):
        """
        Get the primary table tracker.

        Returns:
            TableTracker or None: The primary tracker if one exists, None otherwise
        """
        if not self.trackers:
            return None

        return (
            max(self.trackers, key=lambda tracker: tracker.last_confidence)
            if self.trackers
            else None
        )
