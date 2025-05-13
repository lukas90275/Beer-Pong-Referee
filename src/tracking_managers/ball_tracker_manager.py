import logging
import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from tracking_managers.tracker_manager import TrackerManager


class BallTrackerManager(TrackerManager):
    """
    Specialized tracker manager for ping pong balls.
    Uses Hough Circle Transform for detection and simple centroid tracking.
    """
    REAL_TABLE_WIDTH_FEET = 8.0
    REAL_BALL_DIAMETER_INCHES = 1.57
    CONFIRMATION_FRAMES = 5

    def __init__(
        self,
        max_lost_frames=1,
        hough_dp=1.2,
        hough_min_dist=20,
        hough_param1=50,
        hough_param2=12,
        hough_min_radius=5,
        hough_max_radius=35,
        white_threshold=165,
        min_white_percent=0.08,
        use_blob_detection=True,
        max_elongation_ratio=5.0,        
        max_match_distance=50,
        trajectory_len=10,
        debug_visualization=False,
        **kwargs
    ):
        super().__init__(
            iou_threshold=0.01,
            min_confidence_frames=1,
            max_lost_frames=max_lost_frames,
            detection_threshold=0.01, 
            draw_region_color=(100, 255, 255), 
            draw_region_label_prefix="Ball Region",
            require_regions=False, 
            **kwargs
        )
        
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius
        
        self.white_threshold = white_threshold
        self.min_white_percent = min_white_percent
        
        self.use_blob_detection = use_blob_detection
        self.max_elongation_ratio = max_elongation_ratio
        
        if self.use_blob_detection:
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = white_threshold
            params.maxThreshold = 255
            params.thresholdStep = 10
            params.filterByColor = True
            params.blobColor = 255
            params.filterByArea = True
            params.minArea = 60
            params.maxArea = 3000
            params.filterByCircularity = True
            params.minCircularity = 0.05
            params.filterByConvexity = True
            params.minConvexity = 0.7
            params.filterByInertia = True
            params.minInertiaRatio = 0.01
            self.blob_detector = cv2.SimpleBlobDetector_create(params)
        else:
            self.blob_detector = None
        
        self.expected_ball_diameter_px = None
        self.expected_ball_area_px = None
        
        self.debug_visualization = debug_visualization
        self.logger = logging.getLogger("BallTracker")
        
        self.active_tracks: List[Dict[str, Any]] = []
        self.next_track_id = 0
        self.max_match_distance = max_match_distance
        self.trajectory_len = trajectory_len
        self._last_raw_detections: List[Dict[str, Any]] = []

        self.velocity_smoothing_factor = 0.7

    def detect_circles(self, frame):
        """
        Detect circles in the frame using HoughCircles and blob detection for fast-moving,
        elongated balls caused by motion blur.
        
        Args:
            frame: BGR image
            
        Returns:
            List of detected circles as detections with confidence
        """
        if frame is None or frame.size == 0:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
        
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        circle_detections = self._detect_circles_hough(gray, white_mask, frame.shape[:2], frame)
        
        blob_detections = []
        if self.use_blob_detection:
            blob_detections = self._detect_elongated_blobs(white_mask, frame.shape[:2], frame)
            
        all_detections = circle_detections + blob_detections
        
        if circle_detections and blob_detections:
            all_detections = self._merge_nearby_detections(all_detections)
            
        return all_detections
        
    def _detect_circles_hough(self, gray, white_mask, frame_shape, frame):
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            gray: Grayscale image
            white_mask: Binary mask of white pixels
            frame_shape: Shape of the frame (height, width)
            frame: Original BGR frame for color extraction
            
        Returns:
            List of circle detections
        """
        detections = []
        all_circles = []
        
        circles1 = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius
        )
        
        if circles1 is not None:
            all_circles.extend(circles1[0])
        
        circles2 = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1 * 0.8,
            param2=self.hough_param2 * 0.8,
            minRadius=self.hough_min_radius,
            maxRadius=int(self.hough_max_radius * 1.2)
        )
        
        if circles2 is not None:
            all_circles.extend(circles2[0])
            
        if not all_circles:
            circles3 = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=self.hough_dp,
                minDist=self.hough_min_dist,
                param1=self.hough_param1 * 0.7,
                param2=self.hough_param2 * 0.6,
                minRadius=self.hough_min_radius,
                maxRadius=int(self.hough_max_radius * 1.5)
            )
            
            if circles3 is not None:
                all_circles.extend(circles3[0])
        
        if not all_circles:
            return []
            
        for circle in all_circles:
            x, y, radius = circle
            
            x1 = int(x - radius)
            y1 = int(y - radius)
            x2 = int(x + radius) 
            y2 = int(y + radius)
            
            h, w = frame_shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            circle_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            cv2.circle(circle_mask, 
                      (int((x2-x1)/2), int((y2-y1)/2)), 
                      int(radius), 
                      255, 
                      -1)
            
            white_roi = white_mask[y1:y2, x1:x2]
            
            if white_roi.shape != circle_mask.shape:
                continue
                
            white_pixels = cv2.countNonZero(cv2.bitwise_and(white_roi, circle_mask))
            circle_area = cv2.countNonZero(circle_mask)
            white_percentage = white_pixels / max(1, circle_area) if circle_area > 0 else 0.0
            
            confidence = white_percentage
            
            if confidence >= self.min_white_percent:
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
                mean_color = cv2.mean(frame, mask=mask)[:3]
                
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": max(0.5, confidence),
                    "label": "ball",
                    "radius": float(radius),
                    "center": (float(x), float(y)),
                    "detection_type": "circle",
                    "avg_color": mean_color
                }                
                detections.append(detection)
                
        return detections
        
    def _detect_elongated_blobs(self, white_mask, frame_shape, frame):
        """
        Detect elongated white blobs that could be motion-blurred balls.
        
        Args:
            white_mask: Binary mask of white pixels
            frame_shape: Shape of the frame (height, width)
            frame: Original BGR frame for color extraction
            
        Returns:
            List of blob detections
        """
        h, w = frame_shape
        detections = []
        
        min_blob_area = 20
        max_blob_area = 2000
        
        if self.expected_ball_area_px is not None:
            min_blob_area = max(20, int(self.expected_ball_area_px * 0.3))
            max_blob_area = int(self.expected_ball_area_px * 4.0)
        
        keypoints = self.blob_detector.detect(white_mask)
        
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size
            
            edge_margin = 3
            if (x < edge_margin or x > w - edge_margin or 
                y < edge_margin or y > h - edge_margin):
                continue
                
            radius = size / 2
            x1 = int(x - radius * 1.5)
            y1 = int(y - radius * 1.5)
            x2 = int(x + radius * 1.5)
            y2 = int(y + radius * 1.5)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w-1, x2)
            y2 = min(h-1, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            roi = white_mask[y1:y2, x1:x2]
            
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]
            
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(largest_contour)
            if area < min_blob_area:
                continue
                
            rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            
            (center_x, center_y), (width, height), angle = rect
            
            if width < height:
                width, height = height, width
                
            elongation_ratio = width / max(height, 1)
            
            if elongation_ratio < 1.1 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            min_x = np.min(box_points[:, 0])
            min_y = np.min(box_points[:, 1])
            max_x = np.max(box_points[:, 0])
            max_y = np.max(box_points[:, 1])
            
            min_x += x1
            min_y += y1
            max_x += x1
            max_y += y1
            
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(w-1, max_x)
            max_y = min(h-1, max_y)
            
            if min_x >= max_x or min_y >= max_y:
                continue
                
            confidence_factor = min(elongation_ratio / 2.0, 1.0)
            confidence = 0.6 * confidence_factor
            
            white_count = cv2.countNonZero(white_mask[min_y:max_y, min_x:max_x])
            total_area = (max_x - min_x) * (max_y - min_y)
            white_percent = white_count / max(total_area, 1)
                        
            min_white_percent_required = self.min_white_percent
            if elongation_ratio > 3.0:
                min_white_percent_required *= 0.7
                
            if white_percent >= min_white_percent_required:
                blob_width = max_x - min_x
                blob_height = max_y - min_y
                estimated_radius = (blob_width + blob_height) / 4.0

                detection = {
                    "box": [min_x, min_y, max_x, max_y],
                    "confidence": max(0.5, confidence),
                    "label": "ball",
                    "center": (int(x), int(y)),
                    "elongation_ratio": elongation_ratio,
                    "angle": angle,
                    "detection_type": "blob",
                    "radius": estimated_radius,
                    "avg_color": mean_color
                }
                               
                motion_angle = angle + 90 if width > height else angle
                motion_angle_rad = np.radians(motion_angle)
                motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
                detection["motion_direction"] = motion_dir.tolist()
                    
                detections.append(detection)
                
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_blob_area or area > max_blob_area:
                continue
                
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            if width < height:
                width, height = height, width
                angle += 90
                
            elongation_ratio = width / max(height, 1)
            if elongation_ratio < 1.5 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            center_x, center_y = int(center_x), int(center_y)
            
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            min_x = max(0, np.min(box_points[:, 0]))
            min_y = max(0, np.min(box_points[:, 1]))
            max_x = min(w-1, np.max(box_points[:, 0]))
            max_y = min(h-1, np.max(box_points[:, 1]))
            
            if min_x >= max_x or min_y >= max_y:
                continue
            
            confidence = min(0.5 + (elongation_ratio - 1.5) * 0.1, 0.8)
            
            contour_width = max_x - min_x
            contour_height = max_y - min_y
            estimated_radius_contour = (contour_width + contour_height) / 4.0

            detection = {
                "box": [min_x, min_y, max_x, max_y],
                "confidence": confidence,
                "label": "ball",
                "center": (center_x, center_y),
                "elongation_ratio": elongation_ratio,
                "angle": angle,
                "detection_type": "elongated_contour",
                "radius": estimated_radius_contour
            }
            
            motion_angle_rad = np.radians(angle)
            motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
            detection["motion_direction"] = motion_dir.tolist()
            detections.append(detection)
        
        dilated_mask = cv2.dilate(white_mask, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_blob_area or area > max_blob_area:
                continue
            
            if np.any([cv2.pointPolygonTest(contour, point, False) >= 0 for detection in detections
                     for point in [detection.get("center", (0,0))]]):
                continue
                
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            if width < height:
                width, height = height, width
                angle += 90
                
            elongation_ratio = width / max(height, 1)
            if elongation_ratio < 2.0 or elongation_ratio > self.max_elongation_ratio:
                continue
                
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            min_x = max(0, np.min(box_points[:, 0]))
            min_y = max(0, np.min(box_points[:, 1]))
            max_x = min(w-1, np.max(box_points[:, 0]))
            max_y = min(h-1, np.max(box_points[:, 1]))
            
            if min_x >= max_x or min_y >= max_y:
                continue
                
            roi = white_mask[min_y:max_y, min_x:max_x]
            white_pixels = cv2.countNonZero(roi)
            total_pixels = (max_x - min_x) * (max_y - min_y)
            white_ratio = white_pixels / max(total_pixels, 1)
            
            if white_ratio < self.min_white_percent * 0.5:
                continue
                
            confidence = 0.4 * min(elongation_ratio / 4.0, 1.0)
            
            dilated_width = max_x - min_x
            dilated_height = max_y - min_y
            estimated_radius_dilated = (dilated_width + dilated_height) / 4.0

            detection = {
                "box": [min_x, min_y, max_x, max_y],
                "confidence": confidence,
                "label": "ball",
                "center": (int(center_x), int(center_y)),
                "elongation_ratio": elongation_ratio,
                "angle": angle,
                "detection_type": "dilated_blob",
                "radius": estimated_radius_dilated 
            }
            
            motion_angle_rad = np.radians(angle)
            motion_dir = np.array([np.cos(motion_angle_rad), np.sin(motion_angle_rad)])
            detection["motion_direction"] = motion_dir.tolist()
                
            detections.append(detection)
            
        return detections
        
    def _merge_nearby_detections(self, detections, distance_threshold=30):
        """
        Merge nearby detections, keeping the highest confidence one.
        
        Args:
            detections: List of detection dictionaries
            distance_threshold: Maximum distance to consider detections as duplicates
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
            
        sorted_detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        keep = [True] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if not keep[i]:
                continue
                
            det_i = sorted_detections[i]
            center_i = det_i.get("center", None)
            
            if center_i is None:
                box_i = det_i.get("box", [0, 0, 0, 0])
                center_i = ((box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2)
                
            for j in range(i+1, len(sorted_detections)):
                if not keep[j]:
                    continue
                    
                det_j = sorted_detections[j]
                center_j = det_j.get("center", None)
                
                if center_j is None:
                    box_j = det_j.get("box", [0, 0, 0, 0])
                    center_j = ((box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2)
                    
                distance = math.hypot(center_i[0] - center_j[0], center_i[1] - center_j[1])
                
                if distance < distance_threshold:
                    keep[j] = False
                    
                    if (det_i.get("detection_type", "") == "circle" and 
                        det_j.get("detection_type", "") in ["blob", "elongated_contour"] and
                        "motion_direction" in det_j):
                        det_i["motion_direction"] = det_j["motion_direction"]
                        det_i["is_elongated"] = True
                        det_i["elongation_ratio"] = det_j.get("elongation_ratio", 2.0)
        
        return [det for i, det in enumerate(sorted_detections) if keep[i]]

    def process_ball_detections(self, frame, frame_shape, table_bounds=None, hand_regions: Optional[List[Dict[str, Any]]] = None, cups_tracked=None, **kwargs):
        """
        Process ball detections using simple centroid tracking.
        This method replaces the old process_detr_results.
        
        Args:
            frame: Original frame (required for detect_circles)
            frame_shape: Frame dimensions (height, width)
            table_bounds: Optional table boundaries for size calibration
            hand_regions: Optional list of hand regions (dicts with 'center' and 'radius') for filtering new tracks.
            cups_tracked: Optional list of tracked cups for detecting when a ball falls into a cup.
            **kwargs: Additional arguments (unused for now)
            
        Returns:
            List of active ball track states (dictionaries)
        """ 
        if frame is None:
            return [
                {
                    "id": track["id"],
                    "box": track["box"],
                    "centroid": track["centroid"],
                    "radius": track["radius"],
                    "age": track["age"],
                    "confirmed": track["confirmed"],
                }
                for track in self.active_tracks
            ]
        
        if frame_shape and len(frame_shape) >= 2:
            self.frame_width = frame_shape[1]
        
        if table_bounds is not None:
            self._calibrate_ball_size_from_table(table_bounds, frame_shape)
            
        current_detections = self.detect_circles(frame)
        self._last_raw_detections = current_detections

        num_current_detections = len(current_detections)
        num_active_tracks = len(self.active_tracks)

        predicted_track_centroids = []
        for track in self.active_tracks:
            vx, vy = track.get('velocity', (0.0, 0.0))
            predicted_centroid = (track['centroid'][0] + vx, track['centroid'][1] + vy)
            predicted_track_centroids.append(predicted_centroid)

        matched_indices = []
        used_detection_indices = [False] * num_current_detections

        if num_active_tracks > 0 and num_current_detections > 0:
            dist_matrix = np.full((num_active_tracks, num_current_detections), np.inf)
            for t_idx, track in enumerate(self.active_tracks):
                pred_centroid = predicted_track_centroids[t_idx]
                
                max_distance = self.max_match_distance
                if track.get('is_bouncing', False):
                    max_distance = self.max_match_distance * 2.0
                
                for d_idx, det in enumerate(current_detections):
                    dist = math.hypot(pred_centroid[0] - det['center'][0], 
                                      pred_centroid[1] - det['center'][1])
                    
                    track_radius = track['radius']
                    det_radius = det['radius']
                    if det_radius > track_radius * 2 or det_radius < track_radius / 2:
                        dist = np.inf  
                        
                    track_color = track.get('avg_color', None)
                    det_color = det.get('avg_color', None)
                    if track_color is not None and det_color is not None:
                        color_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(track_color, det_color)))
                        if color_diff > 75:
                            dist = np.inf
                        
                    dist_matrix[t_idx, d_idx] = dist
            
            for _ in range(min(num_active_tracks, num_current_detections)):
                min_val = np.min(dist_matrix)
                
                t_idx_arr, d_idx_arr = np.where(dist_matrix == min_val)
                if len(t_idx_arr) == 0:
                    break 

                t_idx, d_idx = t_idx_arr[0], d_idx_arr[0]
                                
                track = self.active_tracks[t_idx]
                max_distance = self.max_match_distance * (2.0 if track.get('is_bouncing', False) else 1.0)
                
                if min_val > max_distance:
                    dist_matrix[t_idx, :] = np.inf
                    continue
                
                matched_indices.append((t_idx, d_idx))
                used_detection_indices[d_idx] = True
                
                dist_matrix[t_idx, :] = np.inf
                dist_matrix[:, d_idx] = np.inf
        
        updated_tracks = []
        matched_track_indices_set = set()

        for track_idx, detection_idx in matched_indices:
            track = self.active_tracks[track_idx]
            detection = current_detections[detection_idx]
            
            old_centroid = track['centroid']
            new_centroid = detection['center']

            if track.get('is_bouncing', False):
                track['bounced'] = True  

            current_vx = new_centroid[0] - old_centroid[0]
            current_vy = new_centroid[1] - old_centroid[1]
            
            prev_vx, prev_vy = track.get('velocity', (0.0, 0.0))
            smoothed_vx = (prev_vx * (1 - self.velocity_smoothing_factor) + 
                           current_vx * self.velocity_smoothing_factor)
            smoothed_vy = (prev_vy * (1 - self.velocity_smoothing_factor) + 
                           current_vy * self.velocity_smoothing_factor)
            
            track['velocity'] = (smoothed_vx, smoothed_vy)
            track['centroid'] = new_centroid
            track['box'] = detection['box']
            track['radius'] = detection['radius']
            track['age'] = 0
            track['is_bouncing'] = False
            
            track['seen_count'] += 1
            if track['seen_count'] >= self.CONFIRMATION_FRAMES:
                track['confirmed'] = True
                
            track['trajectory'].append(detection['center'])
            if len(track['trajectory']) > self.trajectory_len:
                track['trajectory'].pop(0)
            updated_tracks.append(track)
            matched_track_indices_set.add(track_idx)

        for track_idx, track in enumerate(self.active_tracks):
            if track_idx not in matched_track_indices_set:
                track['age'] += 1    

                if not track['confirmed']:
                    track['seen_count'] = 0
                
                if track['confirmed'] and track['age'] == self.max_lost_frames + 1 and table_bounds:
                    is_bouncing = self._check_if_ball_bouncing(track, cups_tracked, table_bounds)
                    
                    if is_bouncing:
                        track['is_bouncing'] = True
                        updated_tracks.append(track)
                        continue
                    
                if track['age'] <= self.max_lost_frames:
                    updated_tracks.append(track)
                else:
                    if track['confirmed'] and cups_tracked:
                        self._check_if_ball_in_cup(track, cups_tracked)


        for detection_idx, detection in enumerate(current_detections):
            if not used_detection_indices[detection_idx]:
                is_near_hand = False
                ball_center = detection['center']
                for hand_region in hand_regions:
                    hand_center = hand_region['center']
                    hand_radius = hand_region['radius']
                    distance = math.hypot(ball_center[0] - hand_center[0], ball_center[1] - hand_center[1])
                    if distance <= hand_radius:
                        is_near_hand = True
                        break

                if is_near_hand:
                    new_track = {
                        'id': self.next_track_id,
                        'centroid': detection['center'],
                        'box': detection['box'],
                        'radius': detection['radius'],
                        'age': 0,
                        'seen_count': 1,
                        'confirmed': False,
                        'trajectory': [detection['center']],
                        'velocity': (0.0, 0.0),
                        'is_bouncing': False,
                        'bounced': False,
                        'avg_color': detection.get('avg_color', (255, 255, 255))
                    }
                    updated_tracks.append(new_track)
                    self.next_track_id += 1
        
        self.active_tracks = updated_tracks
        
        return [
            {
                "id": track["id"],
                "box": track["box"],
                "centroid": track["centroid"],
                "radius": track["radius"],
                "age": track["age"],
                "velocity": track.get("velocity", (0.0, 0.0)),
                "confirmed": track["confirmed"],
                "is_bouncing": track.get("is_bouncing", False),
                "avg_color": track.get("avg_color", (255, 255, 255))
            }
            for track in self.active_tracks
        ]
        
    def _calibrate_ball_size_from_table(self, table_bounds, frame_shape):
        """
        Calibrate ball size based on table dimensions.
        A standard beer pong table is 8ft long, and a ping pong ball is 1.57 inches.
        
        Args:
            table_bounds: Dictionary with x1, y1, x2, y2 of table
            frame_shape: Shape of the frame (height, width)
        """
        if not table_bounds:
            return
            
        table_width_px = table_bounds["x2"] - table_bounds["x1"]
        if table_width_px <= 0:
            return
            
        pixels_per_inch = table_width_px / (self.REAL_TABLE_WIDTH_FEET * 12)
        
        expected_ball_diameter_px = int(self.REAL_BALL_DIAMETER_INCHES * pixels_per_inch)
        
        expected_ball_area_px = int(3.14159 * (expected_ball_diameter_px / 2) ** 2)
        
        if expected_ball_diameter_px >= 3:
            self.expected_ball_diameter_px = expected_ball_diameter_px
            self.expected_ball_area_px = expected_ball_area_px
            
            self.hough_min_radius = max(2, int(expected_ball_diameter_px * 0.25))
            self.hough_max_radius = int(expected_ball_diameter_px * 0.8)
            
            if self.use_blob_detection and hasattr(self, 'blob_detector') and self.blob_detector is not None:
                params = cv2.SimpleBlobDetector_Params()
                
                params.minThreshold = self.white_threshold
                params.maxThreshold = 255
                params.thresholdStep = 10
                params.filterByColor = True
                params.blobColor = 255
                
                params.filterByArea = True
                min_area = max(20, int(expected_ball_area_px * 0.3))
                max_area = int(expected_ball_area_px * 4.0)
                params.minArea = min_area
                params.maxArea = max_area
                
                params.filterByCircularity = True
                params.minCircularity = 0.05
                params.filterByConvexity = True
                params.minConvexity = 0.7
                params.filterByInertia = True
                params.minInertiaRatio = 0.01
                
                self.blob_detector = cv2.SimpleBlobDetector_create(params)
            
            return True
            
        return False

    def draw_trackers(self, frame, **kwargs):
        """
        Draw all active ball trackers (centroid-based).
        Only confirmed balls will be drawn.
        
        Args:
            frame: Frame to draw on
            **kwargs: Additional arguments (unused)
            
        Returns:
            Frame with trackers drawn
        """
        for track in self.active_tracks:
            if not track.get('confirmed', False):
                continue
                
            centroid = track['centroid']
            radius = track['radius']
            track_id = track['id']
            
            is_bouncing = track.get('is_bouncing', False)
            
            circle_color = (0, 255, 255) if is_bouncing else (0, 255, 0)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), int(radius), circle_color, 2)
            
            id_text = f"ID: {track_id}{' B' if is_bouncing else ''}"
            cv2.putText(
                frame,
                id_text,
                (int(centroid[0] - radius), int(centroid[1] - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                circle_color,
                2
            )

            if self.debug_visualization and 'velocity' in track:
                vx, vy = track['velocity']
                if abs(vx) > 0.1 or abs(vy) > 0.1:
                    start_point = (int(centroid[0]), int(centroid[1]))
                    end_point = (int(centroid[0] + vx * 5), int(centroid[1] + vy * 5))
                    cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 1)

            if self.debug_visualization and len(track['trajectory']) > 1:
                points = np.array(track['trajectory'], dtype=np.int32)
                cv2.polylines(frame, [points.reshape(-1, 1, 2)], False, (255, 255, 0), 1)
                
        return frame
        
    def draw_detections(self, frame, detections):
        """
        Draw circle and blob detections on the frame for debugging.
        
        Args:
            frame: Frame to draw on
            detections: List of detected circles
            
        Returns:
            Frame with detections drawn
        """
        for det in detections:
            det_type = det.get("detection_type", "unknown")
            
            confidence = det.get("confidence", 0.0)
            box = det.get("box", [0, 0, 0, 0])
            
            if det_type == "circle":
                center = det["center"]
                radius = det["radius"]
                
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.3:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                
                cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), color, 2)
                
                cv2.circle(frame, (int(center[0]), int(center[1])), 2, color, -1)
                
            elif det_type in ["blob", "elongated_contour"]:
                if "angle" in det and "elongation_ratio" in det:
                    center = det.get("center", ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
                    angle = det["angle"]
                    elongation = det["elongation_ratio"]
                    
                    elong_factor = min((elongation - 1.0) / 2.0, 1.0)
                    blue = 0
                    green = int(165 * (1 - elong_factor))
                    red = 255
                    color = (blue, green, red)
                    
                    rect = ((center[0], center[1]), 
                           (int(elongation * 20), 20), 
                           angle)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int0(box_points)
                    cv2.drawContours(frame, [box_points], 0, color, 2)
                    
                    if "motion_direction" in det:
                        motion_dir = np.array(det["motion_direction"])
                        start_pt = (int(center[0]), int(center[1]))
                        end_pt = (int(center[0] + motion_dir[0] * 30), 
                                 int(center[1] + motion_dir[1] * 30))
                        cv2.arrowedLine(frame, start_pt, end_pt, (255, 0, 255), 2)
                else:
                    color = (255, 128, 0)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            else:
                color = (0, 128, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
                
        return frame 

    def get_last_raw_detections(self) -> List[Dict[str, Any]]:
        """Returns the raw detections from the last call to detect_circles."""
        return self._last_raw_detections 

    def _check_if_ball_in_cup(self, ball_track, cups_tracked):
        """
        Check if a ball that's about to be removed likely fell into a cup.
        
        Args:
            ball_track: The ball track that's being removed
            cups_tracked: List of tracked cups
            
        Returns:
            bool: True if ball was detected in a cup, False otherwise
        """            
        ball_center = ball_track.get('centroid', None)
        ball_radius = ball_track.get('radius', 0)
        if not ball_center:
            return False
        
        ball_velocity = ball_track.get('velocity', (0, 0))
        velocity_magnitude = math.hypot(ball_velocity[0], ball_velocity[1])
        min_velocity_threshold = 0.0
        
        if velocity_magnitude < min_velocity_threshold:
            return False
            
        for i, cup in enumerate(cups_tracked):
            cup_box = cup.get('box', None)
                
            cup_center = ((cup_box[0] + cup_box[2]) / 2, (cup_box[1] + cup_box[3]) / 2)
            
            dist = math.hypot(ball_center[0] - cup_center[0], ball_center[1] - cup_center[1])
            
            cup_width = cup_box[2] - cup_box[0]
            cup_height = cup_box[3] - cup_box[1]
            cup_radius = min(cup_width, cup_height) / 2
            
            cup_region = cup.get('region_id', None)
            cup_id = cup.get('id', 'unknown')
            
            distance_threshold = cup_radius * 5
            
            if dist < distance_threshold:
                cup_region = cup.get('region_id', None)
                
                if cup_region is None and 'id' in cup:
                    cup_id = cup.get('id', '')
                    if isinstance(cup_id, str):
                        if 'left' in cup_id.lower():
                            cup_region = 'left'
                        elif 'right' in cup_id.lower():
                            cup_region = 'right'
                
                if cup_region is None:
                    frame_center_x = 0
                    if hasattr(self, 'frame_width') and self.frame_width:
                        frame_center_x = self.frame_width / 2
                    else:
                        frame_center_x = 960
                    
                    cup_region = 'left' if cup_center[0] < frame_center_x else 'right'
                
                ball_id = ball_track.get('id', '?')
                
                team = "Unknown"
                if isinstance(cup_region, str):
                    if cup_region == "left" or "left" in cup_region.lower():
                        team = "Left Team"
                    elif cup_region == "right" or "right" in cup_region.lower():
                        team = "Right Team"
                
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                
                bounce_text = ""
                points = 1
                if ball_track.get('bounced', False):
                    bounce_text = "Bounce shot - "
                    points = 2
                
                print(f"🎯 {bounce_text}POINT SCORED! [{timestamp}] Ball (ID: {ball_id}) landed in {team}'s cup! ({points} points)")
                return True
                
        return False

    def _check_if_ball_bouncing(self, track, cups_tracked, table_bounds):
        """
        Check if a ball that's about to be removed might be bouncing on the table or cups.
        Only applied when we've lost normal tracking and the ball should be on/in table bounds.
        
        Args:
            track: The ball track to check
            cups_tracked: List of tracked cups
            table_bounds: Dictionary with x1, y1, x2, y2 of table bounds
            
        Returns:
            bool: True if the ball is likely bouncing, False otherwise
        """
        if not track or not track.get('centroid') or not table_bounds:
            return False
            
        current_x, current_y = track['centroid']
        vx, vy = track.get('velocity', (0, 0))
        ball_radius = track.get('radius', 5) 
        if abs(vy) < 2.0 or vy <= 0:
            return False
        
        predicted_x = current_x + vx
        predicted_y = current_y + vy
        
        bounce_vx = vx
        bounce_vy = -vy * 0.7
        bounce_x = current_x + bounce_vx
        bounce_y = current_y + bounce_vy
        
        
        if table_bounds:
            table_y1 = table_bounds.get('y1', 0)
            table_y2 = table_bounds.get('y2', 0)
            table_x1 = table_bounds.get('x1', 0)
            table_x2 = table_bounds.get('x2', 0)
            
            margin = max(15, ball_radius * 2)
            
            expanded_x1 = table_x1 - margin
            expanded_x2 = table_x2 + margin
            expanded_y1 = table_y1 - margin
            expanded_y2 = table_y1 + margin 
            
            is_horizontally_near_table = expanded_x1 <= current_x <= expanded_x2
            
            is_near_table_surface = expanded_y1 <= current_y <= expanded_y2
            
            is_inside_table = (table_x1 <= current_x <= table_x2) and (table_y1 <= current_y <= table_y2)
            
            crosses_table = (current_y <= table_y1 and predicted_y > table_y1) and is_horizontally_near_table
            
            if (is_near_table_surface and is_horizontally_near_table) or is_inside_table or crosses_table:
                track['velocity'] = (bounce_vx, bounce_vy)
                track['centroid'] = (bounce_x, bounce_y)
                track['age'] = 0
                return True
        
        if cups_tracked:
            for i, cup in enumerate(cups_tracked):
                cup_box = cup.get('box', None)
                if not cup_box:
                    continue
                    
                cup_top = cup_box[1]
                cup_left = cup_box[0]
                cup_right = cup_box[2]
                cup_bottom = cup_box[3]
                cup_width = cup_right - cup_left
                cup_radius = cup_width / 2
                cup_center_x = (cup_left + cup_right) / 2
                
                cup_margin = max(10, ball_radius * 1.5)
                
                horizontal_distance = abs(current_x - cup_center_x)
                is_horizontally_near_cup = horizontal_distance <= (cup_radius + cup_margin)
                
                rim_margin = cup_margin
                is_at_rim_level = abs(current_y - cup_top) <= rim_margin
                
                is_inside_cup = (cup_left <= current_x <= cup_right) and (cup_top <= current_y <= cup_bottom)
                
                crosses_rim = (current_y <= cup_top and predicted_y > cup_top) and is_horizontally_near_cup
                
                if is_at_rim_level and is_horizontally_near_cup or is_inside_cup or crosses_rim:
                    bounce_vy = -vy * 0.6
                    track['velocity'] = (bounce_vx, bounce_vy)
                    track['centroid'] = (bounce_x, current_y + bounce_vy)
                    track['age'] = 0
                    return True
                         
        return False 