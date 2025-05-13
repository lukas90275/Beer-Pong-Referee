from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from tracking_managers.ball_tracker_manager import BallTrackerManager
from tracking_managers.cup_tracker_manager import CupTrackerManager
from tracking_managers.skeleton_tracker_manager import SkeletonTrackerManager
from tracking_managers.table_tracker_manager import TableTrackerManager

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

skeleton_tracker = SkeletonTrackerManager(model_path='pose_landmarker.task', max_poses=4)

tracker_managers = {
    "table": TableTrackerManager(),
    "cup": CupTrackerManager(),
    "ball": BallTrackerManager(),
}

frame_counter = 0

prev_hands_tracked: Dict[str, Dict] = {}
prev_elbows_tracked: Dict[str, Dict] = {}
handled_violations: Set[Tuple[str, int]] = set()
last_ball_possession: Dict[int, str] = {}


def overlay_transparent(background, overlay, alpha):
    """
    Add a transparent overlay on the background image
    """
    output = background.copy()
    cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0, output)
    return output


def check_elbow_rule_violations(current_hands: Dict[str, Dict], 
                              current_elbows: Dict[str, Dict], 
                              table_bounds: Optional[Dict[str, int]]) -> None:
    """
    Check for elbow rule violations when balls leave hands.
    
    Args:
        current_hands: Current frame's tracked hands
        current_elbows: Current frame's tracked elbows
        table_bounds: Dict with x1, y1, x2, y2 keys for table bounds or None
    """
    global prev_hands_tracked, prev_elbows_tracked, handled_violations, last_ball_possession
    
    if not table_bounds or "x1" not in table_bounds or "x2" not in table_bounds:
        return
    
    table_x1, table_x2 = table_bounds["x1"], table_bounds["x2"]
    
    for hand_id, prev_hand_data in prev_hands_tracked.items():
        prev_balls = set(prev_hand_data.get("balls_inside", []))
        
        current_balls = set(current_hands.get(hand_id, {}).get("balls_inside", []))
        
        balls_left_hand = prev_balls - current_balls
        
        for ball_id in balls_left_hand:
            violation_key = (hand_id, ball_id)
            if violation_key in handled_violations:
                continue
            
            last_ball_possession[ball_id] = hand_id
            
            elbow_id = f"{hand_id}_ELBOW"
            
            if elbow_id in prev_elbows_tracked:
                elbow_pos = prev_elbows_tracked[elbow_id].get("position")
                if elbow_pos:
                    elbow_x = elbow_pos[0]
                    
                    if table_x1 <= elbow_x <= table_x2:
                        side = prev_elbows_tracked[elbow_id].get("side", "UNKNOWN")
                        person_id = prev_elbows_tracked[elbow_id].get("person_id", "?")
                        
                        print(f"Elbow Rule Violation: Player {person_id}'s {side} elbow was over the table when throwing. "
                              f"Elbow X: {elbow_x}, Table bounds: [{table_x1}, {table_x2}]")
                        
                        handled_violations.add(violation_key)
    
    all_current_balls = set()
    for hand_data in current_hands.values():
        all_current_balls.update(hand_data.get("balls_inside", []))
    
    handled_violations_to_remove = set()
    for hand_id, ball_id in handled_violations:
        if ball_id in all_current_balls:
            handled_violations_to_remove.add((hand_id, ball_id))
    
    handled_violations -= handled_violations_to_remove


def analyze_frame(frame, table_viz=True, cup_viz=True, ball_viz=True, 
                 cup_search_viz=False, table_search_viz=False, ball_search_viz=True,
                 pose_viz=True, hand_viz=True, elbow_viz=True):
    """
    Analyze a frame using DETR to detect objects of interest in beer pong.

    Args:
        frame: numpy array of the image in BGR format (from cv2)
        table_viz: whether to visualize table detection (default: True)
        cup_viz: whether to visualize cup detection (default: True)
        ball_viz: whether to visualize ball detection (default: True)
        cup_search_viz: whether to visualize cup search boxes (default: False)
        table_search_viz: whether to visualize table search boxes (default: False)
        ball_search_viz: whether to visualize ball search/detection regions (default: True)
        pose_viz: whether to visualize human pose detection (default: True)
        hand_viz: whether to visualize hand tracking (default: True)
        elbow_viz: whether to visualize elbow tracking (default: True)

    Returns:
        annotated_frame: frame with bounding boxes and labels
        detections: dictionary containing detection information
    """
    global frame_counter, tracker_managers, skeleton_tracker
    global prev_hands_tracked, prev_elbows_tracked
    frame_counter += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb_frame, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([frame.shape[:2]])

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.1,
    )[0]

    detections = {
        "table_tracker": None,
        "cups_tracked": [],
        "balls_tracked": [],
        "poses_tracked": None,
        "hands_tracked": {},
        "elbows_tracked": {}
    }

    annotated_frame = frame.copy()
    
    detections["table_tracker"] = tracker_managers["table"].process_detr_results(
        results, model, frame.shape[:2]
    )
    
    table_bounds_dict = None
    table_tracker = tracker_managers["table"].get_primary_tracker()
    if table_tracker and hasattr(table_tracker, 'box') and table_tracker.box is not None:
        bounds = table_tracker.box
        table_bounds_dict = {
            "x1": bounds[0],
            "y1": bounds[1],
            "x2": bounds[2],
            "y2": bounds[3],
        }

    detections["cups_tracked"] = tracker_managers["cup"].process_detr_results(
        results, model, frame.shape[:2], table_bounds=table_bounds_dict
    )
        
    hand_regions_for_ball_tracking = []
    if pose_viz or hand_viz or elbow_viz:
        pose_result = skeleton_tracker.process_frame(frame)
        detections["poses_tracked"] = pose_result
        if pose_result:
            detections["hands_tracked"] = skeleton_tracker.get_hands()
            detections["elbows_tracked"] = skeleton_tracker.get_elbows()
            for hand_id, hand_data in detections["hands_tracked"].items():
                if hand_data and 'position' in hand_data and 'radius' in hand_data:
                    hand_regions_for_ball_tracking.append({
                        'center': hand_data['position'],
                        'radius': hand_data['radius']
                    })

    detections["balls_tracked"] = tracker_managers["ball"].process_ball_detections(
        frame, frame.shape[:2], 
        table_bounds=table_bounds_dict,
        hand_regions=hand_regions_for_ball_tracking,
        cups_tracked=detections["cups_tracked"]
    )
    
    if detections["balls_tracked"] and detections["hands_tracked"]:
        skeleton_tracker.update_balls_in_hands(detections["balls_tracked"])
    
    if prev_hands_tracked and prev_elbows_tracked:
        check_elbow_rule_violations(
            detections["hands_tracked"],
            detections["elbows_tracked"],
            table_bounds_dict
        )

    if table_viz and table_tracker:
        annotated_frame = tracker_managers["table"].draw_trackers(annotated_frame, show_search_box=table_search_viz)
    
    if cup_viz:
        if cup_search_viz:
            annotated_frame = tracker_managers["cup"].draw_regions(annotated_frame, show_expected_size=True)
        annotated_frame = tracker_managers["cup"].draw_trackers(annotated_frame, show_search_box=cup_search_viz)
        
    if ball_viz:
        annotated_frame = tracker_managers["ball"].draw_trackers(annotated_frame)

    if pose_viz and detections["poses_tracked"]:
        annotated_frame = skeleton_tracker.draw_landmarks_on_image(
            annotated_frame, 
            detections["poses_tracked"],
            draw_hands=hand_viz,
            draw_elbows=elbow_viz
        )
    
    prev_hands_tracked = deepcopy(detections["hands_tracked"])
    prev_elbows_tracked = deepcopy(detections["elbows_tracked"])

    return annotated_frame, detections
