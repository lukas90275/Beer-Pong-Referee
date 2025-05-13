import uuid
from abc import ABC

import cv2
import numpy as np


class TrackerConfig:
    """Configuration object for ObjectTracker to reduce parameter explosion"""
    
    def __init__(
        self,
        tracker_type="Object",
        color=(255, 0, 0),
        min_confidence_frames=3,
        max_lost_frames=5,
        search_expansion_factor=1.5,
        velocity_smoothing=0.7,
        velocity_threshold=1.0,
        position_stability_factor=0.0,
        use_x_distance_only=True,
        trajectory_len=0,
        detect_aspect_ratio_change=False,
        aspect_ratio_change_factor=1.5,
        aspect_ratio_history_len=5,
        search_region_bounds=None,
        source_region_id=None,
        directional_search_expansion=False,
    ):
        self.tracker_type = tracker_type
        self.color = color
        
        self.min_confidence_frames = min_confidence_frames
        self.max_lost_frames = max_lost_frames
        self.search_expansion_factor = search_expansion_factor
        
        self.velocity_smoothing = velocity_smoothing
        self.velocity_threshold = velocity_threshold
        self.position_stability_factor = position_stability_factor
        self.use_x_distance_only = use_x_distance_only
        self.directional_search_expansion = directional_search_expansion
        
        self.trajectory_len = trajectory_len
        self.detect_aspect_ratio_change = detect_aspect_ratio_change
        self.aspect_ratio_change_factor = aspect_ratio_change_factor 
        self.aspect_ratio_history_len = aspect_ratio_history_len
        self.search_region_bounds = search_region_bounds
        self.source_region_id = source_region_id


class ObjectTracker(ABC):
    """
    Unified tracking class for all object types.
    Manages state like box, velocity, confidence, and lost frames.
    Supports specialized features through configuration and overridable methods.
    """

    def __init__(
        self,
        initial_box,
        frame_shape,
        tracker_id=None,
        initial_confidence=1.0,
        config=None,
        **kwargs
    ):
        self.config = config if config is not None else TrackerConfig(**kwargs)
        
        self.id = tracker_id if tracker_id is not None else uuid.uuid4()
        self.box = np.array(initial_box, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.raw_velocity = np.zeros(2, dtype=float)
        self.center = self._calculate_center(self.box)
        self.prev_center = self.center.copy()
        self.lost_frames = 0
        self.confidence_frames = 1
        self.is_confident = False
        self.is_lost = False
        self.frame_shape = frame_shape
        self.search_box = self.box.copy()
        self.last_confidence = initial_confidence
        
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]

        self.position_history = []
        self.max_position_history = 5
        self.position_history.append(self.center.copy())
        
        self.trajectory_positions = []
        if self.config.trajectory_len > 0:
            self.trajectory_positions.append(self.center.copy())
            
        self.is_standing = True
        if self.config.detect_aspect_ratio_change:
            self.original_aspect_ratio = self.width / self.height if self.height > 0 else 1.0
            self.aspect_ratio_history = [self.original_aspect_ratio]

        if self.confidence_frames >= self.config.min_confidence_frames:
            self.is_confident = True

    def _calculate_center(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _update_box_from_center(self, center, width, height):
        x1 = center[0] - width / 2
        y1 = center[1] - height / 2
        x2 = center[0] + width / 2
        y2 = center[1] + height / 2
        return np.array([x1, y1, x2, y2])
        
    def calculate_distance(self, point_a, point_b):
        """
        Calculate distance between two points, using only x-direction if configured.
        Can be scaled by object size if needed.
        
        Args:
            point_a: First point as (x, y) or [x, y]
            point_b: Second point as (x, y) or [x, y]
            
        Returns:
            Distance (float): Euclidean or x-only distance
        """
        if self.config.use_x_distance_only:
            return abs(point_a[0] - point_b[0])
        else:
            return np.linalg.norm(np.array(point_a) - np.array(point_b))
    
    def calculate_scaled_distance(self, point_a, point_b):
        """
        Calculate distance scaled by object size (width).
        
        Args:
            point_a: First point as (x, y) or [x, y]
            point_b: Second point as (x, y) or [x, y]
            
        Returns:
            Scaled distance (float): Distance scaled by object width
        """
        raw_distance = self.calculate_distance(point_a, point_b)
        
        if self.width < 1.0:
            return raw_distance
            
        return raw_distance / self.width

    def _clip_to_frame(self, box):
        """Clip box coordinates to stay within frame boundaries."""
        h, w = self.frame_shape[:2]
        box[0] = np.clip(box[0], 0, w - 1)
        box[1] = np.clip(box[1], 0, h - 1)
        box[2] = np.clip(box[2], 0, w - 1)
        box[3] = np.clip(box[3], 0, h - 1)
        if box[0] >= box[2]:
            box[2] = box[0] + 1
        if box[1] >= box[3]:
            box[3] = box[1] + 1
        box[2] = np.clip(box[2], 0, w - 1)
        box[3] = np.clip(box[3], 0, h - 1)
        return box
        
    def _clip_to_bounds(self, box, bounds):
        """Clip box coordinates to stay within specified bounds."""
        if bounds is None:
            return box
        box[0] = np.clip(box[0], bounds[0], bounds[2])
        box[1] = np.clip(box[1], bounds[1], bounds[3])
        box[2] = np.clip(box[2], bounds[0], bounds[2])
        box[3] = np.clip(box[3], bounds[1], bounds[3])
        if box[0] >= box[2]: box[2] = box[0] + 1
        if box[1] >= box[3]: box[3] = box[1] + 1
        box[2] = np.clip(box[2], bounds[0], bounds[2]) 
        box[3] = np.clip(box[3], bounds[1], bounds[3])
        return box

    def predict(self):
        """
        Predict the bounding box and search area for the next frame based on velocity.
        The search box is expanded based on lost frames.
        Returns the predicted box (current self.box) and the search box.
        """
        velocity_scale = 1.0
        if self.lost_frames > 0:
            velocity_scale = min(self.lost_frames, 3)

        predicted_center = self.center + (self.velocity * velocity_scale)

        width = self.box[2] - self.box[0]
        height = self.box[3] - self.box[1]
        self.box = self._update_box_from_center(predicted_center, width, height)
        self.center = predicted_center
        
        self.width = width
        self.height = height
        
        if self.config.position_stability_factor > 0 and len(self.position_history) >= 2:
            recent_motion = 0
            if self.config.use_x_distance_only:
                recent_motion = abs(self.position_history[-1][0] - self.position_history[0][0])
            else:
                recent_motion = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            
            motion_threshold = min(5.0, self.width * 0.2)
            
            if recent_motion < motion_threshold:
                stable_center = np.mean(self.position_history, axis=0)
                
                stabilized_center = (
                    self.center * (1 - self.config.position_stability_factor) + 
                    stable_center * self.config.position_stability_factor
                )
                
                self.box = self._update_box_from_center(stabilized_center, width, height)
                
                self.center = stabilized_center

        if self.lost_frames > 0:
            base_expansion = self.config.search_expansion_factor * self.lost_frames
        else:
            base_expansion = self.config.search_expansion_factor
            
        if self.config.directional_search_expansion and abs(self.velocity[0]) > self.config.velocity_threshold:
            direction_x = np.sign(self.velocity[0])
            
            if direction_x > 0:
                left_expansion = width * base_expansion
                right_expansion = width * base_expansion * 2
            else:
                left_expansion = width * base_expansion * 2
                right_expansion = width * base_expansion
            
            top_expansion = height * base_expansion
            bottom_expansion = height * base_expansion
        else:
            left_expansion = width * base_expansion
            right_expansion = width * base_expansion
            top_expansion = height * base_expansion
            bottom_expansion = height * base_expansion
        
        self.search_box = np.array([
            self.center[0] - left_expansion,
            self.center[1] - top_expansion,
            self.center[0] + right_expansion,
            self.center[1] + bottom_expansion
        ])

        self.box = self._clip_to_frame(self.box)
        self.search_box = self._clip_to_frame(self.search_box)
        
        if self.config.search_region_bounds is not None:
            self.search_box = self._clip_to_bounds(
                self.search_box, self.config.search_region_bounds
            )
            
        self._custom_predict()

        return self.box.astype(int), self.search_box.astype(int)
        
    def _custom_predict(self):
        """Hook for subclasses to add custom prediction logic without overriding the full predict method"""
        pass

    def update(self, detection_box, detection_confidence=1.0, **kwargs):
        """
        Update the tracker state with a new assigned detection.
        """
        detection_box = np.array(detection_box, dtype=float)
        new_center = self._calculate_center(detection_box)

        self.prev_center = self.center.copy()

        if self.config.use_x_distance_only:
            self.raw_velocity[0] = new_center[0] - self.center[0]
            self.raw_velocity[1] = (new_center[1] - self.center[1]) * 0.5
        else:
            self.raw_velocity = new_center - self.center

        self.velocity = self.velocity * self.config.velocity_smoothing + self.raw_velocity * (
            1 - self.config.velocity_smoothing
        )

        mask = np.abs(self.velocity) < self.config.velocity_threshold
        self.velocity[mask] = 0.0

        self.box = detection_box
        self.center = new_center
        self.lost_frames = 0
        self.is_lost = False
        self.confidence_frames += 1
        self.last_confidence = detection_confidence
        
        self.width = self.box[2] - self.box[0]
        self.height = self.box[3] - self.box[1]
        
        if self.confidence_frames >= self.config.min_confidence_frames:
            self.is_confident = True
            
        self.position_history.append(self.center.copy())
        if len(self.position_history) > self.max_position_history:
            self.position_history.pop(0)
            
        if self.config.trajectory_len > 0:
            self.trajectory_positions.append(self.center.copy())
            if len(self.trajectory_positions) > self.config.trajectory_len:
                self.trajectory_positions.pop(0)
                
        if self.config.detect_aspect_ratio_change and self.height > 0:
            current_aspect = self.width / self.height
            self.aspect_ratio_history.append(current_aspect)
            
            if len(self.aspect_ratio_history) > self.config.aspect_ratio_history_len:
                self.aspect_ratio_history.pop(0)
                
            if len(self.aspect_ratio_history) >= 3:
                avg_recent_aspect = np.mean(self.aspect_ratio_history[-3:])
                
                if self.is_standing and avg_recent_aspect > self.original_aspect_ratio * self.config.aspect_ratio_change_factor:
                    self.is_standing = False
                elif not self.is_standing and abs(avg_recent_aspect - self.original_aspect_ratio) < 0.2:
                    self.is_standing = True
            
        self._custom_update(**kwargs)

    def _custom_update(self, **kwargs):
        """Hook for subclasses to add custom update logic without overriding the full update method"""
        pass

    def mark_lost(self):
        """
        Mark the object as lost for this frame.
        """
        self.lost_frames += 1
        self.confidence_frames = 0
        self.is_confident = False

        self.velocity *= 0.9

        if self.lost_frames >= self.config.max_lost_frames:
            self.is_lost = True

    def match_score(self, detection_box):
        """
        Calculate a score indicating how well the detection matches this tracker's current state.
        When the object is lost, use the search_box instead of the box for matching to allow
        for velocity-based prediction and expanding search area.
        """
        if self.lost_frames > 0:
            detection_center = self._calculate_center(detection_box)
            
            if self.config.use_x_distance_only:
                if not (self.search_box[0] <= detection_center[0] <= self.search_box[2]):
                    return 0.0
                if not (self.search_box[1] - self.height <= detection_center[1] <= self.search_box[3] + self.height):
                    return 0.0
            else:
                if not (
                    self.search_box[0] <= detection_center[0] <= self.search_box[2]
                    and self.search_box[1] <= detection_center[1] <= self.search_box[3]
                ):
                    return 0.0
                    
            return self.calculate_iou(self.box, detection_box)
            
        return self.calculate_iou(self.box, detection_box)

    def get_state(self):
        """
        Return the current state of the tracker.
        Includes all base fields and any enabled specialized fields.
        """
        state = {
            "id": self.id,
            "box": self.box.astype(int).tolist(),
            "search_box": self.search_box.astype(int).tolist(),
            "center": self.center.astype(int).tolist(),
            "velocity": self.velocity.tolist(),
            "lost_frames": self.lost_frames,
            "confidence_frames": self.confidence_frames,
            "is_confident": self.is_confident,
            "is_lost": self.is_lost,
            "last_confidence": self.last_confidence,
            "tracker_type": self.config.tracker_type,
            "width": self.width,
            "height": self.height,
            "diameter": max(self.width, self.height),
        }
        
        if self.config.source_region_id is not None:
            state["source_region_id"] = self.config.source_region_id
            
        if self.config.detect_aspect_ratio_change:
            state["is_standing"] = self.is_standing
            if len(self.aspect_ratio_history) > 0:
                state["aspect_ratio"] = self.aspect_ratio_history[-1]
            
        self._extend_state(state)
            
        return state
        
    def _extend_state(self, state):
        """Hook for subclasses to extend the state without overriding the full get_state method"""
        pass

    def draw(self, frame, show_search_box=False):
        """
        Draw the tracker's state onto the frame.
        Handles drawing all enabled specialized visualizations.
        """
        if show_search_box:
            sb = self.search_box.astype(int)
            cv2.rectangle(
                frame, (sb[0], sb[1]), (sb[2], sb[3]), (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"{self.config.tracker_type} Search",
                (sb[0], sb[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        b = self.box.astype(int)
        color = self.config.color if self.lost_frames == 0 else (0, 0, 255)
        thickness = 2 
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)

        cv2.putText(
            frame,
            f"{self.config.tracker_type}",
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
        if show_search_box and self.config.trajectory_len > 0 and len(self.trajectory_positions) >= 2:
            for i in range(1, len(self.trajectory_positions)):
                pt1 = self.trajectory_positions[i-1].astype(int)
                pt2 = self.trajectory_positions[i].astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 200, 255), 2)
                
        self._custom_draw(frame, show_search_box)

        return frame
        
    def _custom_draw(self, frame, show_search_box):
        """Hook for subclasses to add custom drawing without overriding the full draw method"""
        pass

    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = (
            interArea / float(boxAArea + boxBArea - interArea)
            if (boxAArea + boxBArea - interArea) > 0
            else 0
        )

        return iou
