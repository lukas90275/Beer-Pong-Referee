import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SkeletonTrackerManager:
    """
    Manages detection and visualization of human poses using MediaPipe Pose Landmarker.
    Tracks up to 4 people simultaneously.
    """

    HAND_LANDMARKS = {
        'LEFT_WRIST': 15,
        'LEFT_THUMB': 17,
        'LEFT_INDEX': 19,
        'LEFT_PINKY': 21,
        'RIGHT_WRIST': 16, 
        'RIGHT_THUMB': 18,
        'RIGHT_INDEX': 20,
        'RIGHT_PINKY': 22
    }

    ELBOW_LANDMARKS = {
        'LEFT_ELBOW': 13,
        'RIGHT_ELBOW': 14
    }

    BODY_LANDMARKS = {
        'NOSE': 0,
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24
    }
    BODY_CENTER_LANDMARK_NAMES = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']


    def __init__(self, model_path='pose_landmarker.task', max_poses=4, hand_interaction_radius=40):
        """
        Initialize the skeleton tracker manager.
        
        Args:
            model_path: Path to the MediaPipe pose landmarker model
            max_poses: Maximum number of poses to detect (default: 4)
            hand_interaction_radius: Radius in pixels to consider for hand interaction (default: 30)
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=max_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        self.tracked_hands = {}
        self.tracked_elbows = {}
        
        self.hand_interaction_radius = hand_interaction_radius

        self.tracked_persons = {}
        self.next_person_id = 0
        self.person_tracking_max_unseen_frames = 10
        self.person_tracking_distance_threshold = 0.25
        
        self.frame_count = 0
        self.min_frames_for_tracking = 3

    def _calculate_pose_center(self, pose_landmarks):
        """
        Calculate the center of a pose using specified body landmarks.
        Args:
            pose_landmarks: A list of MediaPipe NormalizedLandmark objects.
        Returns:
            A tuple (center_x, center_y) in normalized coordinates, or None if center cannot be calculated.
        """
        coords = []
        for name in self.BODY_CENTER_LANDMARK_NAMES:
            idx = self.BODY_LANDMARKS.get(name)
            if idx is not None and idx < len(pose_landmarks):
                landmark = pose_landmarks[idx]
                if landmark.visibility > 0.5:
                    coords.append((landmark.x, landmark.y))
        
        if len(coords) >= 2:
            center_x = sum(c[0] for c in coords) / len(coords)
            center_y = sum(c[1] for c in coords) / len(coords)
            return (center_x, center_y)
        return None

    def _normalized_distance(self, p1, p2):
        """Calculate Euclidean distance between two normalized 2D points."""
        if p1 is None or p2 is None:
            return float('inf')
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_frame(self, frame):
        """
        Process a frame to detect poses.
        
        Args:
            frame: CV2 BGR image
            
        Returns:
            pose_result: MediaPipe PoseLandmarkerResult or None if no poses detected
        """
        self.frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        pose_result = self.pose_detector.detect(mp_image)
        
        if pose_result.pose_landmarks:
            if self.frame_count > self.min_frames_for_tracking:
                self._update_person_and_hand_tracking(pose_result.pose_landmarks, frame.shape[1], frame.shape[0])
            else:
                self.tracked_hands = {}
                self.tracked_elbows = {}
        else:
            self.tracked_hands = {}
            self.tracked_elbows = {}
            
        return pose_result if pose_result.pose_landmarks else None
        
    def _update_person_and_hand_tracking(self, current_poses_landmarks_list, frame_width, frame_height):
        """
        Update tracked persons and their hands based on current pose detection results.
        Assigns stable IDs to persons and derives stable IDs for hands.
        
        Args:
            current_poses_landmarks_list: List of pose landmarks from MediaPipe for the current frame.
            frame_width: Width of the input frame.
            frame_height: Height of the input frame.
        """
        current_poses_info = []
        for i, landmarks in enumerate(current_poses_landmarks_list):
            center = self._calculate_pose_center(landmarks)
            if center:
                current_poses_info.append({
                    'original_idx': i, 
                    'landmarks': landmarks, 
                    'center': center, 
                    'matched_person_id': None
                })

        for person_id in list(self.tracked_persons.keys()):
            self.tracked_persons[person_id]['frames_unseen'] += 1

        sorted_tracked_person_ids = sorted(self.tracked_persons.keys())
        matched_current_pose_indices = [False] * len(current_poses_info)

        for person_id in sorted_tracked_person_ids:
            person_data = self.tracked_persons[person_id]
            best_match_idx = -1
            min_dist = self.person_tracking_distance_threshold

            for i, current_pose in enumerate(current_poses_info):
                if not matched_current_pose_indices[i]:
                    dist = self._normalized_distance(person_data['center'], current_pose['center'])
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = i
            
            if best_match_idx != -1:
                matched_pose_info = current_poses_info[best_match_idx]
                self.tracked_persons[person_id]['landmarks'] = matched_pose_info['landmarks']
                self.tracked_persons[person_id]['center'] = matched_pose_info['center']
                self.tracked_persons[person_id]['frames_unseen'] = 0
                matched_pose_info['matched_person_id'] = person_id
                matched_current_pose_indices[best_match_idx] = True
        
        for i, current_pose in enumerate(current_poses_info):
            if not matched_current_pose_indices[i] and current_pose['matched_person_id'] is None:
                new_pid = self.next_person_id
                self.tracked_persons[new_pid] = {
                    'landmarks': current_pose['landmarks'],
                    'center': current_pose['center'],
                    'frames_unseen': 0
                }
                current_pose['matched_person_id'] = new_pid
                self.next_person_id += 1

        pids_to_remove = [
            pid for pid, p_data in self.tracked_persons.items() 
            if p_data['frames_unseen'] > self.person_tracking_max_unseen_frames
        ]
        for pid in pids_to_remove:
            del self.tracked_persons[pid]

        new_tracked_hands = {}
        for current_pose in current_poses_info:
            person_id = current_pose['matched_person_id']
            if person_id is None:
                continue

            pose_landmarks = current_pose['landmarks']
            
            for side in ['LEFT', 'RIGHT']:
                wrist_idx = self.HAND_LANDMARKS[f'{side}_WRIST']
                index_idx = self.HAND_LANDMARKS[f'{side}_INDEX']
                thumb_idx = self.HAND_LANDMARKS[f'{side}_THUMB']

                if index_idx < len(pose_landmarks) and thumb_idx < len(pose_landmarks):
                    index = pose_landmarks[index_idx]
                    thumb = pose_landmarks[thumb_idx]

                    if index.visibility > 0.5 and thumb.visibility > 0.5:
                        hand_x = int((thumb.x + index.x) / 2 * frame_width)
                        hand_y = int((thumb.y + index.y) / 2 * frame_height)
                        
                        hand_id = f"P{person_id}_{side[0]}"
                        
                        new_tracked_hands[hand_id] = {
                            'id': hand_id,
                            'position': (hand_x, hand_y),
                            'side': side,
                            'person_id': person_id,
                            'radius': self.hand_interaction_radius,
                            'balls_inside': []
                        }
        self.tracked_hands = new_tracked_hands

        new_tracked_elbows = {}
        for current_pose in current_poses_info:
            person_id = current_pose['matched_person_id']
            if person_id is None:
                continue
            
            pose_landmarks = current_pose['landmarks']

            for side in ['LEFT', 'RIGHT']:
                elbow_landmark_name = f'{side}_ELBOW'
                elbow_idx = self.ELBOW_LANDMARKS.get(elbow_landmark_name)

                if elbow_idx is not None and elbow_idx < len(pose_landmarks):
                    elbow_landmark = pose_landmarks[elbow_idx]

                    if elbow_landmark.visibility > 0.5:
                        elbow_x = int(elbow_landmark.x * frame_width)
                        elbow_y = int(elbow_landmark.y * frame_height)
                        
                        elbow_id = f"P{person_id}_{side[0]}_ELBOW"
                        
                        new_tracked_elbows[elbow_id] = {
                            'id': elbow_id,
                            'position': (elbow_x, elbow_y),
                            'side': side,
                            'person_id': person_id
                        }
        self.tracked_elbows = new_tracked_elbows
    
    def get_hands(self):
        """
        Get the dictionary of tracked hands.
        
        Returns:
            dict: Dictionary of tracked hands
        """
        return self.tracked_hands
        
    def get_elbows(self):
        """
        Get the dictionary of tracked elbows.
        
        Returns:
            dict: Dictionary of tracked elbows
        """
        return self.tracked_elbows
        
    def update_balls_in_hands(self, active_ball_tracks):
        """
        Update the list of balls considered to be inside each tracked hand.

        Args:
            active_ball_tracks: A list of dictionaries, where each dictionary represents an
                                active ball track and must contain 'id', 'centroid', and 'radius'.
                                Ball tracks should also include 'confirmed' status.
        """
        if not active_ball_tracks:
            for hand_id in self.tracked_hands:
                self.tracked_hands[hand_id]['balls_inside'] = []
            return

        for hand_id, hand_data in self.tracked_hands.items():
            hand_data['balls_inside'] = []
            hand_center_x, hand_center_y = hand_data['position']
            hand_radius = hand_data['radius']

            for ball_track in active_ball_tracks:
                ball_id = ball_track.get('id')
                ball_centroid = ball_track.get('centroid')
                ball_radius = ball_track.get('radius', 0)
                ball_confirmed = ball_track.get('confirmed', False)

                if ball_id is None or ball_centroid is None:
                    continue
                    
                if not ball_confirmed:
                    continue

                ball_center_x, ball_center_y = ball_centroid

                distance = np.sqrt((hand_center_x - ball_center_x)**2 + (hand_center_y - ball_center_y)**2)

                if distance < hand_radius:
                    hand_data['balls_inside'].append(ball_id)

    def draw_landmarks_on_image(self, frame, detection_result, draw_hands=True, draw_elbows=True):
        """
        Draw pose landmarks on the image.
        
        Args:
            frame: CV2 BGR image
            detection_result: MediaPipe PoseLandmarkerResult
            draw_hands: Whether to draw circles around detected hands
            draw_elbows: Whether to draw circles around detected elbows
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if not detection_result or not detection_result.pose_landmarks:
            return frame
            
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame.copy()
        
        pose_landmarks_list = detection_result.pose_landmarks
        
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                rgb_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else rgb_image

        if draw_hands:
            for hand_info in self.tracked_hands.values():
                position = hand_info['position']
                side = hand_info['side']
                radius = hand_info['radius']
                color = (0, 255, 0) if side == 'LEFT' else (0, 0, 255)
                
                cv2.circle(output_image, position, radius, color, 2)
                
                cv2.putText(output_image, f"ID: {hand_info['id']}", 
                           (position[0] - 20, position[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                balls_count = len(hand_info.get('balls_inside', []))
                if balls_count > 0:
                    cv2.putText(output_image, f"Balls: {balls_count}",
                               (position[0] - 20, position[1] - radius - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if draw_elbows:
            for elbow_info in self.tracked_elbows.values():
                position = elbow_info['position']
                side = elbow_info['side']
                
                color = (255, 165, 0) if side == 'LEFT' else (128, 0, 128)
                elbow_radius = 10
                
                cv2.circle(output_image, position, elbow_radius, color, 2)
                
                cv2.putText(output_image, f"ID: {elbow_info['id']}", 
                           (position[0] - 20, position[1] - elbow_radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return output_image 
        
        if not draw_hands and not draw_elbows:
             return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else rgb_image
        elif not draw_elbows:
            return output_image
        
        return output_image
        
    def get_pose_count(self, detection_result):
        """
        Get the number of poses detected.
        
        Args:
            detection_result: MediaPipe PoseLandmarkerResult
            
        Returns:
            int: Number of poses detected
        """
        if not detection_result or not detection_result.pose_landmarks:
            return 0
        return len(detection_result.pose_landmarks)
        
    def release(self):
        """Release resources used by the pose detector."""
        self.tracked_persons.clear()
        self.tracked_hands.clear()
        self.tracked_elbows.clear()
        self.next_person_id = 0       
