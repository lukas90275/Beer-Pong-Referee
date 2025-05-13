import cv2
import numpy as np

from trackers.cup_tracker import CupTracker
from tracking_managers.tracker_manager import TrackerManager


class CupTrackerManager(TrackerManager):
    """
    Manager for tracking multiple cups on a beer pong table.
    Uses table bounds to focus detection on the ends of the table.
    Filters detections based on expected size derived from table dimensions.
    """
    REAL_TABLE_WIDTH_FEET = 8.0
    REAL_CUP_HEIGHT_INCHES = 4.85
    REAL_CUP_WIDTH_INCHES = 3.84

    def __init__(
        self,
        iou_threshold=0.6,
        min_confidence_frames=5,
        max_lost_frames=15,
        detection_threshold=0.1,
        size_tolerance=0.5,
        **kwargs
    ):
        self.size_tolerance = size_tolerance
        self.left_region = None
        self.right_region = None
        self.pixels_per_foot = None
        self.expected_cup_height_pixels = None
        self.expected_cup_width_pixels = None
        
        super().__init__(
            iou_threshold=iou_threshold,
            min_confidence_frames=min_confidence_frames,
            max_lost_frames=max_lost_frames,
            detection_threshold=detection_threshold,
            track_single_instance=False,
            motion_history_max_len=5,
            target_labels=["cup", "wine glass", "bottle"],
            tracker_overlap_threshold=0.6,
            min_tracker_separation=0.01,
            cost_weight_iou=0.4,
            cost_weight_distance=0.6,
            position_stability_factor=0.95,
            position_id_distance_threshold=20,
            position_id_prefer_x_distance=True,
            position_id_name_format="{region}_cup_{index}",
            region_calibration=self._calibrate_cup_regions,
            draw_region_color=(0, 255, 255),
            draw_region_label_prefix="Cup Region",
            require_regions=True,
            high_confidence_fallback=True,
            high_confidence_threshold=0.65,
            **kwargs
        )
        
        self.position_history["left"] = []
        self.position_history["right"] = []
        
        self.register_tracker_factory(self._create_cup_tracker)
    
    def _create_tracker_from_detection(self, detection, frame_shape):
        """Create a new tracker from detection - implements abstract method"""
        return self._create_cup_tracker(detection, frame_shape)
    
    def _create_cup_tracker(self, detection, frame_shape):
        """Create a new cup tracker from a detection."""
        box = self._extract_bbox_from_detection(detection)
        confidence = detection.get("confidence", 0.0)
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        region_bounds = None
        region_name = self.get_region_for_point(center_x, center_y)
            
        if region_name == "left" and self.left_region:
            region_bounds = [
                self.left_region["x1"],
                self.left_region["y1"],
                self.left_region["x2"],
                self.left_region["y2"],
            ]
        elif region_name == "right" and self.right_region:
            region_bounds = [
                self.right_region["x1"],
                self.right_region["y1"],
                self.right_region["x2"],
                self.right_region["y2"],
            ]
            
        cup_id = self.get_stable_id_for_position(region_name, center_x, center_y, "cup")
            
        return CupTracker(
            initial_box=box,
            frame_shape=frame_shape,
            initial_confidence=confidence,
            search_region_bounds=region_bounds,
            tracker_id=cup_id,
            position_stability_factor=self.position_stability_factor,
        )

    def _calibrate_cup_regions(self, table_bounds=None, **_):
        """
        Calibrate the left and right cup regions based on the table bounds.
        Also estimates expected cup size based on table width.
        
        Args:
            table_bounds: Dictionary with x1, y1, x2, y2 coordinates of the table
        """
        if not table_bounds:
            return
            
        table_width_pixels = table_bounds["x2"] - table_bounds["x1"]
        table_height_pixels = table_bounds["y2"] - table_bounds["y1"]
        
        self.pixels_per_foot = table_width_pixels / self.REAL_TABLE_WIDTH_FEET if table_width_pixels > 0 else None
        
        if self.pixels_per_foot:
            self.expected_cup_height_pixels = (
                self.REAL_CUP_HEIGHT_INCHES / 12.0
            ) * self.pixels_per_foot
            self.expected_cup_width_pixels = (
                self.REAL_CUP_WIDTH_INCHES / 12.0
            ) * self.pixels_per_foot
            
            self.size_min_width = self.expected_cup_width_pixels * (1 - self.size_tolerance)
            self.size_max_width = self.expected_cup_width_pixels * (1 + self.size_tolerance)
            self.size_min_height = self.expected_cup_height_pixels * (1 - self.size_tolerance)
            self.size_max_height = self.expected_cup_height_pixels * (1 + self.size_tolerance)
            
            expected_aspect_ratio = self.expected_cup_width_pixels / self.expected_cup_height_pixels
            self.size_min_aspect_ratio = expected_aspect_ratio * 0.5
            self.size_max_aspect_ratio = expected_aspect_ratio * 1.5
            
            self.position_id_distance_threshold = self.expected_cup_width_pixels * 0.5
            
        region_width = int(table_width_pixels * 0.3)
        
        region_start_y = int(table_bounds["y1"] - table_height_pixels * 0.5)
        region_end_y = int(table_bounds["y1"] + table_height_pixels * 0.85)
        
        self.left_region = {
            "x1": table_bounds["x1"],
            "x2": table_bounds["x1"] + region_width,
            "y1": max(0, region_start_y),
            "y2": region_end_y,
        }
        
        self.right_region = {
            "x1": table_bounds["x2"] - region_width,
            "x2": table_bounds["x2"],
            "y1": max(0, region_start_y),
            "y2": region_end_y,
        }
        
        self.clear_regions()
        
        if self.left_region:
            self.add_region([
                self.left_region["x1"],
                self.left_region["y1"],
                self.left_region["x2"],
                self.left_region["y2"],
            ])
            
        if self.right_region:
            self.add_region([
                self.right_region["x1"],
                self.right_region["y1"],
                self.right_region["x2"],
                self.right_region["y2"],
            ])
    
    def get_region_for_point(self, x, y):
        """
        Override of base method to return named regions ("left"/"right") instead of UUIDs.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            region_name: "left", "right", or None
        """
        if self.left_region and (self.left_region["x1"] <= x <= self.left_region["x2"] and 
                                self.left_region["y1"] <= y <= self.left_region["y2"]):
            return "left"
        elif self.right_region and (self.right_region["x1"] <= x <= self.right_region["x2"] and 
                                  self.right_region["y1"] <= y <= self.right_region["y2"]):
            return "right"
        return None
        
    def draw_regions(self, frame, **kwargs):
        """
        Draw cup search regions and display expected size info on the frame.
        """
        frame = super().draw_regions(frame, **kwargs)
        
        if self.expected_cup_width_pixels and self.expected_cup_height_pixels:
            size_text = f"Exp Cup Size (px): {self.expected_cup_width_pixels:.1f}W x {self.expected_cup_height_pixels:.1f}H"
            cv2.putText(frame, size_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        return frame
        
    def update(self, detections, frame_shape, table_bounds=None, **kwargs):
        """
        Update cup trackers with new detections, optionally calibrating with table bounds.
        Legacy method for backward compatibility.
        """
        return super().update(detections, frame_shape, table_bounds=table_bounds, **kwargs)
        
    def process_detr_results(self, results, model, frame_shape, frame=None, table_bounds=None, **kwargs):
        """
        Process DETR detection results, optionally calibrating with table bounds.
        Legacy method for backward compatibility.
        """
        return super().process_detr_results(results, model, frame_shape, frame=frame, table_bounds=table_bounds, **kwargs) 