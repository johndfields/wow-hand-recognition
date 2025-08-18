"""
Simple implementation of hand gesture recognition for backward compatibility.
This module provides a simplified version of the hand gesture recognition system
that is compatible with the original simple implementation's configuration format.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class SimpleGestureRecognizer:
    """Simple implementation of hand gesture recognition."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the simple gesture recognizer.
        
        Args:
            config_path: Path to configuration file
        """
        # Default configuration
        self.config = {
            "camera_index": 0,
            "camera_width": 960,
            "camera_height": 540,
            "min_detection_confidence": 0.6,
            "min_tracking_confidence": 0.6,
            "stable_frames": 3,
            "sensitivity": 1.0,
            "cooldown": 0.7,
            "hand": "right",
            "mappings": {}
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config["min_detection_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"]
        )
        
        # Initialize state
        self.last_gesture = None
        self.last_gesture_time = 0
        self.stable_gesture_count = 0
        self.current_stable_gesture = None
        
        logger.info("Simple gesture recognizer initialized")
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def process_frame(self, frame: Any) -> Tuple[Any, Optional[str]]:
        """
        Process a frame and detect gestures.
        
        Args:
            frame: The frame to process
            
        Returns:
            Tuple of (processed frame, detected gesture)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect gesture
                gesture = self._detect_gesture(hand_landmarks.landmark)
                
                # Stabilize gesture detection
                if gesture == self.last_gesture:
                    self.stable_gesture_count += 1
                else:
                    self.stable_gesture_count = 0
                    self.last_gesture = gesture
                
                # Check if gesture is stable
                if self.stable_gesture_count >= self.config["stable_frames"]:
                    # Check cooldown
                    current_time = time.time()
                    if (current_time - self.last_gesture_time > self.config["cooldown"] and 
                        gesture != self.current_stable_gesture):
                        self.current_stable_gesture = gesture
                        self.last_gesture_time = current_time
                        
                        # Add gesture name to frame
                        if gesture:
                            cv2.putText(
                                frame,
                                f"Gesture: {gesture}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                2
                            )
                        
                        return frame, gesture
        
        return frame, None
    
    def _detect_gesture(self, landmarks: List[Any]) -> Optional[str]:
        """
        Detect gesture from landmarks.
        
        Args:
            landmarks: List of landmark points
            
        Returns:
            Gesture name or None
        """
        # Check for open palm
        if self._is_open_palm(landmarks):
            return "open_palm"
        
        # Check for fist
        if self._is_fist(landmarks):
            return "fist"
        
        # Check for thumbs up
        if self._is_thumbs_up(landmarks):
            return "thumbs_up"
        
        # Check for one finger
        if self._is_one_finger(landmarks):
            return "one_finger"
        
        # Check for two fingers
        if self._is_two_fingers(landmarks):
            return "two_fingers"
        
        # Check for three fingers
        if self._is_three_fingers(landmarks):
            return "three_fingers"
        
        # Check for four fingers
        if self._is_four_fingers(landmarks):
            return "four_fingers"
        
        # Check for L shape
        if self._is_l_shape(landmarks):
            return "l_shape"
        
        # Check for hang loose
        if self._is_hang_loose(landmarks):
            return "hang_loose"
        
        return None
    
    def _is_open_palm(self, landmarks: List[Any]) -> bool:
        """Check if hand is in open palm gesture."""
        # All fingers extended
        return (self._is_finger_extended(landmarks, 8, 5) and  # Index
                self._is_finger_extended(landmarks, 12, 9) and  # Middle
                self._is_finger_extended(landmarks, 16, 13) and  # Ring
                self._is_finger_extended(landmarks, 20, 17))  # Pinky
    
    def _is_fist(self, landmarks: List[Any]) -> bool:
        """Check if hand is in fist gesture."""
        # All fingers curled
        return (not self._is_finger_extended(landmarks, 8, 5) and  # Index
                not self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17) and  # Pinky
                not self._is_thumb_extended(landmarks))  # Thumb
    
    def _is_thumbs_up(self, landmarks: List[Any]) -> bool:
        """Check if hand is in thumbs up gesture."""
        # Thumb extended, others curled
        return (self._is_thumb_extended(landmarks) and
                not self._is_finger_extended(landmarks, 8, 5) and  # Index
                not self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17))  # Pinky
    
    def _is_one_finger(self, landmarks: List[Any]) -> bool:
        """Check if hand is showing one finger (index)."""
        # Index extended, others curled
        return (self._is_finger_extended(landmarks, 8, 5) and  # Index
                not self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17) and  # Pinky
                not self._is_thumb_extended(landmarks))  # Thumb
    
    def _is_two_fingers(self, landmarks: List[Any]) -> bool:
        """Check if hand is showing two fingers (index and middle)."""
        # Index and middle extended, others curled
        return (self._is_finger_extended(landmarks, 8, 5) and  # Index
                self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17) and  # Pinky
                not self._is_thumb_extended(landmarks))  # Thumb
    
    def _is_three_fingers(self, landmarks: List[Any]) -> bool:
        """Check if hand is showing three fingers (index, middle, ring)."""
        # Index, middle, and ring extended, others curled
        return (self._is_finger_extended(landmarks, 8, 5) and  # Index
                self._is_finger_extended(landmarks, 12, 9) and  # Middle
                self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17) and  # Pinky
                not self._is_thumb_extended(landmarks))  # Thumb
    
    def _is_four_fingers(self, landmarks: List[Any]) -> bool:
        """Check if hand is showing four fingers (all except thumb)."""
        # All fingers except thumb extended
        return (self._is_finger_extended(landmarks, 8, 5) and  # Index
                self._is_finger_extended(landmarks, 12, 9) and  # Middle
                self._is_finger_extended(landmarks, 16, 13) and  # Ring
                self._is_finger_extended(landmarks, 20, 17) and  # Pinky
                not self._is_thumb_extended(landmarks))  # Thumb
    
    def _is_l_shape(self, landmarks: List[Any]) -> bool:
        """Check if hand is in L shape (thumb and index extended)."""
        # Thumb and index extended, others curled
        return (self._is_thumb_extended(landmarks) and
                self._is_finger_extended(landmarks, 8, 5) and  # Index
                not self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                not self._is_finger_extended(landmarks, 20, 17))  # Pinky
    
    def _is_hang_loose(self, landmarks: List[Any]) -> bool:
        """Check if hand is in hang loose gesture (thumb and pinky extended)."""
        # Thumb and pinky extended, others curled
        return (self._is_thumb_extended(landmarks) and
                not self._is_finger_extended(landmarks, 8, 5) and  # Index
                not self._is_finger_extended(landmarks, 12, 9) and  # Middle
                not self._is_finger_extended(landmarks, 16, 13) and  # Ring
                self._is_finger_extended(landmarks, 20, 17))  # Pinky
    
    def _is_finger_extended(self, landmarks: List[Any], tip_idx: int, pip_idx: int) -> bool:
        """
        Check if a finger is extended.
        
        Args:
            landmarks: List of landmark points
            tip_idx: Index of the finger tip
            pip_idx: Index of the PIP joint (second joint)
            
        Returns:
            True if finger is extended, False otherwise
        """
        sensitivity = self.config["sensitivity"]
        
        # Get coordinates
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        wrist = landmarks[0]
        
        # Calculate distances
        tip_to_wrist = self._distance(tip, wrist)
        pip_to_wrist = self._distance(pip, wrist)
        
        # Finger is extended if tip is further from wrist than PIP
        return tip_to_wrist > pip_to_wrist * (1.1 / sensitivity)
    
    def _is_thumb_extended(self, landmarks: List[Any]) -> bool:
        """Check if thumb is extended."""
        sensitivity = self.config["sensitivity"]
        
        # Get coordinates
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]
        
        # Calculate distances
        tip_to_wrist = self._distance(thumb_tip, wrist)
        ip_to_wrist = self._distance(thumb_ip, wrist)
        
        # Thumb is extended if tip is further from wrist than IP
        return tip_to_wrist > ip_to_wrist * (1.1 / sensitivity)
    
    def _distance(self, p1: Any, p2: Any) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    
    def execute_action(self, gesture: str) -> bool:
        """
        Execute action for detected gesture.
        
        Args:
            gesture: Detected gesture
            
        Returns:
            True if action was executed, False otherwise
        """
        if not gesture or gesture not in self.config["mappings"]:
            return False
        
        mapping = self.config["mappings"][gesture]
        action = mapping.get("action", "")
        target = mapping.get("target", "")
        mode = mapping.get("mode", "tap")
        
        logger.info(f"Executing action for gesture '{gesture}': {action} {target} ({mode})")
        
        # In the simple implementation, we just log the action
        # The actual input handling is done in the modular implementation
        return True


def main(config: Optional[str] = None, camera: int = 0, width: int = 960, height: int = 540,
         debug: bool = False, no_preview: bool = False):
    """
    Main function for the simple implementation.
    
    Args:
        config: Path to configuration file
        camera: Camera index
        width: Camera width
        height: Camera height
        debug: Enable debug mode
        no_preview: Disable preview
    """
    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Initialize recognizer
    recognizer = SimpleGestureRecognizer(config)
    
    # Override camera settings if provided
    if camera is not None:
        recognizer.config["camera_index"] = camera
    if width is not None:
        recognizer.config["camera_width"] = width
    if height is not None:
        recognizer.config["camera_height"] = height
    
    # Initialize camera
    cap = cv2.VideoCapture(recognizer.config["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, recognizer.config["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, recognizer.config["camera_height"])
    
    # Create window if preview is enabled
    if not no_preview:
        cv2.namedWindow("Simple Hand Gesture Recognition", cv2.WINDOW_NORMAL)
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Process frame
            processed_frame, gesture = recognizer.process_frame(frame)
            
            # Execute action if gesture detected
            if gesture:
                recognizer.execute_action(gesture)
            
            # Update FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Add FPS to frame
            cv2.putText(
                processed_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show frame if preview is enabled
            if not no_preview:
                cv2.imshow("Simple Hand Gesture Recognition", processed_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Clean up
        cap.release()
        if not no_preview:
            cv2.destroyAllWindows()
        
        logger.info("Simple implementation stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Hand Gesture Recognition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=960, help="Camera width")
    parser.add_argument("--height", type=int, default=540, help="Camera height")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview")
    
    args = parser.parse_args()
    
    main(
        config=args.config,
        camera=args.camera,
        width=args.width,
        height=args.height,
        debug=args.debug,
        no_preview=args.no_preview
    )

