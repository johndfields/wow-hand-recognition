#!/usr/bin/env python3
"""
Hand to Key - Simplified Hand Gesture Control System
Focused purely on hand tracking and gesture recognition.
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller as KeyController, Key
from pynput.mouse import Controller as MouseController, Button as MouseButton
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Supported gesture types."""
    OPEN_PALM = "open_palm"
    FIST = "fist"
    ONE_FINGER = "one_finger"  # Only index finger up
    TWO_FINGERS = "two_fingers"  # Index and middle up (victory/peace)
    THREE_FINGERS = "three_fingers"  # Index, middle, ring up
    FOUR_FINGERS = "four_fingers"  # Index, middle, ring, pinky up (no thumb)
    THUMBS_UP = "thumbs_up"
    L_SHAPE = "l_shape"  # L gesture with thumb and index
    HANG_LOOSE = "hang_loose"  # Thumb and pinky extended (shaka)
    PINCH_INDEX = "pinch_index"
    PINCH_MIDDLE = "pinch_middle"
    PINCH_RING = "pinch_ring"
    PINCH_PINKY = "pinch_pinky"


@dataclass
class GestureConfig:
    """Configuration for a gesture mapping."""
    gesture: str
    action: str  # "key" or "mouse"
    target: str
    mode: str = "tap"  # "tap" or "hold"


class HandGestureDetector:
    """Detects hand gestures from MediaPipe landmarks."""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
    
    def detect_gestures(self, landmarks) -> Set[str]:
        """Detect all active gestures from hand landmarks."""
        gestures = set()
        
        # Check pinch gestures first (they take priority)
        pinch_detected = False
        if self._is_pinch(landmarks, 8):
            gestures.add(GestureType.PINCH_INDEX.value)
            pinch_detected = True
        if self._is_pinch(landmarks, 12):
            gestures.add(GestureType.PINCH_MIDDLE.value)
            pinch_detected = True
        if self._is_pinch(landmarks, 16):
            gestures.add(GestureType.PINCH_RING.value)
            pinch_detected = True
        if self._is_pinch(landmarks, 20):
            gestures.add(GestureType.PINCH_PINKY.value)
            pinch_detected = True
        
        # If pinch is detected, don't check other gestures to avoid conflicts
        if pinch_detected:
            return gestures
        
        # Check finger counting gestures (mutually exclusive)
        if self._is_one_finger(landmarks):
            gestures.add(GestureType.ONE_FINGER.value)
        elif self._is_two_fingers(landmarks):
            gestures.add(GestureType.TWO_FINGERS.value)
        elif self._is_three_fingers(landmarks):
            gestures.add(GestureType.THREE_FINGERS.value)
        elif self._is_four_fingers(landmarks):
            gestures.add(GestureType.FOUR_FINGERS.value)
        elif self._is_open_palm(landmarks):
            gestures.add(GestureType.OPEN_PALM.value)
        elif self._is_fist(landmarks):
            gestures.add(GestureType.FIST.value)
        elif self._is_thumbs_up(landmarks):
            gestures.add(GestureType.THUMBS_UP.value)
        elif self._is_l_shape(landmarks):
            gestures.add(GestureType.L_SHAPE.value)
        elif self._is_hang_loose(landmarks):
            gestures.add(GestureType.HANG_LOOSE.value)
        
        return gestures
    
    def _l2_distance(self, p1, p2) -> float:
        """Calculate L2 distance between two landmarks."""
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    
    def _bbox_diag(self, landmarks) -> float:
        """Calculate diagonal of bounding box."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    
    def _finger_extended(self, landmarks, tip_idx: int, pip_idx: int, wrist_idx: int = 0) -> bool:
        """Check if a finger is extended."""
        ratio = 1.15 / self.sensitivity
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        wrist = landmarks[wrist_idx]
        return self._l2_distance(tip, wrist) > self._l2_distance(pip, wrist) * ratio
    
    def _thumb_extended(self, landmarks) -> bool:
        """Check if thumb is extended."""
        return self._finger_extended(landmarks, 4, 3, 0)
    
    def _is_open_palm(self, landmarks) -> bool:
        """Detect open palm gesture (all four fingers + thumb extended)."""
        return all([
            self._finger_extended(landmarks, 8, 6),   # Index
            self._finger_extended(landmarks, 12, 10), # Middle
            self._finger_extended(landmarks, 16, 14), # Ring
            self._finger_extended(landmarks, 20, 18), # Pinky
            self._thumb_extended(landmarks)           # Thumb
        ])
    
    def _is_fist(self, landmarks) -> bool:
        """Detect fist gesture."""
        return not any([
            self._finger_extended(landmarks, 8, 6),
            self._finger_extended(landmarks, 12, 10),
            self._finger_extended(landmarks, 16, 14),
            self._finger_extended(landmarks, 20, 18),
            self._thumb_extended(landmarks)
        ])
    
    def _is_one_finger(self, landmarks) -> bool:
        """Detect one finger up (index only)."""
        return (self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18) and
                not self._thumb_extended(landmarks))
    
    def _is_two_fingers(self, landmarks) -> bool:
        """Detect two fingers up (index and middle)."""
        return (self._finger_extended(landmarks, 8, 6) and 
                self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18) and
                not self._thumb_extended(landmarks))
    
    def _is_three_fingers(self, landmarks) -> bool:
        """Detect three fingers up (index, middle, ring)."""
        return (self._finger_extended(landmarks, 8, 6) and 
                self._finger_extended(landmarks, 12, 10) and
                self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18) and
                not self._thumb_extended(landmarks))
    
    def _is_four_fingers(self, landmarks) -> bool:
        """Detect four fingers up (index, middle, ring, pinky - no thumb)."""
        return (self._finger_extended(landmarks, 8, 6) and 
                self._finger_extended(landmarks, 12, 10) and
                self._finger_extended(landmarks, 16, 14) and
                self._finger_extended(landmarks, 20, 18) and
                not self._thumb_extended(landmarks))
    
    def _is_thumbs_up(self, landmarks) -> bool:
        """Detect thumbs up gesture."""
        return (self._thumb_extended(landmarks) and
                not self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18))
    
    def _is_l_shape(self, landmarks) -> bool:
        """Detect L shape gesture (thumb out, index up, others closed)."""
        # Thumb should be extended to the side
        thumb_extended = self._thumb_extended(landmarks)
        # Index finger should be extended up
        index_extended = self._finger_extended(landmarks, 8, 6)
        # Other fingers should be closed
        middle_closed = not self._finger_extended(landmarks, 12, 10)
        ring_closed = not self._finger_extended(landmarks, 16, 14)
        pinky_closed = not self._finger_extended(landmarks, 20, 18)
        
        # Check if thumb is roughly perpendicular to index (L shape)
        if thumb_extended and index_extended and middle_closed and ring_closed and pinky_closed:
            # Additional check: thumb tip should be away from index
            thumb_tip = landmarks[4]
            index_base = landmarks[5]
            distance = self._l2_distance(thumb_tip, index_base)
            diag = self._bbox_diag(landmarks)
            return distance > 0.15 * diag  # Thumb is extended away
        return False
    
    def _is_hang_loose(self, landmarks) -> bool:
        """Detect hang loose/shaka gesture (thumb and pinky extended, others closed)."""
        # Thumb should be extended
        thumb_extended = self._thumb_extended(landmarks)
        # Pinky should be extended
        pinky_extended = self._finger_extended(landmarks, 20, 18)
        # Other fingers should be closed
        index_closed = not self._finger_extended(landmarks, 8, 6)
        middle_closed = not self._finger_extended(landmarks, 12, 10)
        ring_closed = not self._finger_extended(landmarks, 16, 14)
        
        return (thumb_extended and pinky_extended and 
                index_closed and middle_closed and ring_closed)
    
    def _is_pinch(self, landmarks, tip_idx: int) -> bool:
        """Detect pinch gesture with specific finger."""
        ratio = 0.08 * self.sensitivity
        d = self._l2_distance(landmarks[4], landmarks[tip_idx])
        diag = self._bbox_diag(landmarks) + 1e-6
        return d < ratio * diag


class InputController:
    """Handles keyboard and mouse input."""
    
    def __init__(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()
        self.held_keys: Dict[str, bool] = {}
        
    def execute_action(self, config: GestureConfig):
        """Execute an action based on gesture configuration."""
        if config.action == "key":
            self._handle_key(config)
        elif config.action == "mouse":
            self._handle_mouse(config)
    
    def _handle_key(self, config: GestureConfig):
        """Handle keyboard input."""
        key = self._parse_key(config.target)
        if not key:
            return
        
        if config.mode == "tap":
            self.keyboard.press(key)
            self.keyboard.release(key)
        elif config.mode == "hold":
            if config.target not in self.held_keys:
                self.held_keys[config.target] = False
            
            if not self.held_keys[config.target]:
                self.keyboard.press(key)
                self.held_keys[config.target] = True
    
    def release_key(self, target: str):
        """Release a held key."""
        if target in self.held_keys and self.held_keys[target]:
            key = self._parse_key(target)
            if key:
                self.keyboard.release(key)
                self.held_keys[target] = False
    
    def _handle_mouse(self, config: GestureConfig):
        """Handle mouse input."""
        if config.target == "left_click":
            self.mouse.click(MouseButton.left)
        elif config.target == "right_click":
            self.mouse.click(MouseButton.right)
    
    def _parse_key(self, key_str: str):
        """Parse a key string to a Key object or KeyCode."""
        key_map = {
            'space': Key.space,
            'tab': Key.tab,
            'enter': Key.enter,
            'esc': Key.esc,
            'escape': Key.esc,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'cmd': Key.cmd
        }
        
        key_lower = key_str.lower()
        if key_lower in key_map:
            return key_map[key_lower]
        elif len(key_str) == 1:
            # For digits, use enhanced Parallels-compatible handling
            if key_str.isdigit():
                return self._create_digit_key(key_str)
            else:
                # Use KeyCode for single characters for better Windows compatibility
                from pynput.keyboard import KeyCode
                return KeyCode.from_char(key_str)
        else:
            logger.warning(f"Unknown key: {key_str}")
            return None
    
    def _create_digit_key(self, digit: str):
        """Create a digit key with enhanced VM compatibility (e.g., for Parallels)."""
        from pynput.keyboard import KeyCode
        import platform
        
        # On Mac (which might be sending to VMs like Parallels), use virtual key codes for digits
        if platform.system() == 'Darwin':
            mac_digit_vk_codes = {
                '0': 0x1D,  # kVK_ANSI_0
                '1': 0x12,  # kVK_ANSI_1
                '2': 0x13,  # kVK_ANSI_2
                '3': 0x14,  # kVK_ANSI_3
                '4': 0x15,  # kVK_ANSI_4
                '5': 0x17,  # kVK_ANSI_5
                '6': 0x16,  # kVK_ANSI_6
                '7': 0x1A,  # kVK_ANSI_7
                '8': 0x1C,  # kVK_ANSI_8
                '9': 0x19   # kVK_ANSI_9
            }
            
            if digit in mac_digit_vk_codes:
                try:
                    # Create KeyCode with both virtual key code and character for best compatibility
                    return KeyCode.from_vk(mac_digit_vk_codes[digit], char=digit)
                except Exception as e:
                    logger.debug(f"VK approach failed for digit {digit}: {e}")
        
        # Fallback to character-based KeyCode
        from pynput.keyboard import KeyCode
        return KeyCode.from_char(digit)
    
    def release_all(self):
        """Release all held keys."""
        for target in list(self.held_keys.keys()):
            self.release_key(target)


class HandToKeyApp:
    """Main application for hand gesture control."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.detector = HandGestureDetector(sensitivity=self.config.get('sensitivity', 1.0))
        self.controller = InputController()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Determine max hands based on hand preference
        self.hand_preference = self.config.get('hand', 'any')  # 'left', 'right', or 'any'
        max_hands = 2 if self.hand_preference == 'any' else 1
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=self.config.get('min_detection_confidence', 0.6),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.6)
        )
        
        # State tracking
        self.active_gestures: Set[str] = set()
        self.gesture_cooldowns: Dict[str, float] = {}
        self.is_paused = False
        self.stable_frames: Dict[str, int] = {}
        self.required_stable_frames = self.config.get('stable_frames', 3)
        
        # Camera setup
        self.cap = cv2.VideoCapture(self.config.get('camera_index', 0))
        width = self.config.get('camera_width', 960)
        height = self.config.get('camera_height', 540)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        logger.info("Hand to Key application initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'camera_index': 0,
            'camera_width': 960,
            'camera_height': 540,
            'min_detection_confidence': 0.6,
            'min_tracking_confidence': 0.6,
            'stable_frames': 3,
            'sensitivity': 1.0,
            'cooldown': 0.5,
            'hand': 'any',  # 'left', 'right', or 'any'
            'mappings': {
                'open_palm': {'action': 'key', 'target': 'w', 'mode': 'hold'},
                'fist': {'action': 'key', 'target': 's', 'mode': 'hold'},
                'l_shape': {'action': 'key', 'target': 'a', 'mode': 'hold'},  # L shape for left
                'hang_loose': {'action': 'key', 'target': 'd', 'mode': 'hold'},  # Hang loose for right
                'one_finger': {'action': 'mouse', 'target': 'right_click', 'mode': 'tap'},
                'two_fingers': {'action': 'key', 'target': 'a', 'mode': 'hold'},
                'three_fingers': {'action': 'key', 'target': 'd', 'mode': 'hold'},
                'thumbs_up': {'action': 'key', 'target': 'tab', 'mode': 'tap'},
                'pinch_index': {'action': 'key', 'target': '1', 'mode': 'tap'},
                'pinch_middle': {'action': 'key', 'target': '2', 'mode': 'tap'},
                'pinch_ring': {'action': 'key', 'target': '3', 'mode': 'tap'},
                'pinch_pinky': {'action': 'key', 'target': '4', 'mode': 'tap'}
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return default_config
    
    def run(self):
        """Main application loop."""
        logger.info("Starting Hand to Key application")
        logger.info("Press 'q' to quit, 'p' to pause/resume")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    logger.error("Failed to read from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame if not paused
                if not self.is_paused:
                    self._process_frame(frame)
                
                # Draw UI
                self._draw_ui(frame)
                
                # Display frame
                cv2.imshow('Hand to Key', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.controller.release_all()
                    logger.info(f"{'Paused' if self.is_paused else 'Resumed'}")
        
        finally:
            self.cleanup()
    
    def _process_frame(self, frame):
        """Process a single frame for gesture detection."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Track which gestures are detected this frame
        current_gestures = set()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, results.multi_handedness)):
                
                # Get hand label from MediaPipe
                # In the mirrored view, MediaPipe correctly identifies hands
                detected_hand = handedness.classification[0].label.lower()
                
                # Skip if hand preference is set and doesn't match
                if self.hand_preference != 'any':
                    if detected_hand != self.hand_preference:
                        continue
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect gestures
                detected = self.detector.detect_gestures(hand_landmarks.landmark)
                current_gestures.update(detected)
        
        # Update stable frame counts
        for gesture in self.config['mappings'].keys():
            if gesture in current_gestures:
                self.stable_frames[gesture] = self.stable_frames.get(gesture, 0) + 1
            else:
                self.stable_frames[gesture] = 0
        
        # Process stable gestures
        for gesture in current_gestures:
            if self.stable_frames.get(gesture, 0) >= self.required_stable_frames:
                self._handle_gesture(gesture)
                # Also check for alternative gesture names for backwards compatibility
                if gesture == 'three_fingers' and 'three' in self.config['mappings']:
                    self._handle_gesture('three')
        
        # Release hold keys for gestures that are no longer active
        for gesture in self.active_gestures - current_gestures:
            if gesture in self.config['mappings']:
                mapping = self.config['mappings'][gesture]
                if mapping['mode'] == 'hold':
                    self.controller.release_key(mapping['target'])
        
        # Update active gestures
        self.active_gestures = current_gestures
    
    def _handle_gesture(self, gesture: str):
        """Handle a detected gesture."""
        if gesture not in self.config['mappings']:
            return
        
        mapping = self.config['mappings'][gesture]
        current_time = time.time()
        
        # Check cooldown for tap actions
        if mapping['mode'] == 'tap':
            last_time = self.gesture_cooldowns.get(gesture, 0)
            if current_time - last_time < self.config.get('cooldown', 0.5):
                return
            self.gesture_cooldowns[gesture] = current_time
        
        # Execute action
        config = GestureConfig(
            gesture=gesture,
            action=mapping['action'],
            target=mapping['target'],
            mode=mapping['mode']
        )
        self.controller.execute_action(config)
    
    def _draw_ui(self, frame):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        
        # Status text
        status = "PAUSED" if self.is_paused else "ACTIVE"
        color = (0, 0, 255) if self.is_paused else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Hand preference indicator
        hand_text = f"Hand: {self.hand_preference.upper()}"
        cv2.putText(frame, hand_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Keybindings panel on the right side
        panel_x = w - 220
        cv2.putText(frame, "Keybindings:", (panel_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_offset = 55
        gesture_display_names = {
            'open_palm': 'Open Palm',
            'fist': 'Fist',
            'l_shape': 'L Shape',
            'hang_loose': 'Hang Loose 🤙',
            'one_finger': 'One Finger',
            'two_fingers': 'Two Fingers',
            'three_fingers': 'Three Fingers',
            'four_fingers': 'Four Fingers',
            'thumbs_up': 'Thumbs Up',
            'pinch_index': 'Pinch Index',
            'pinch_middle': 'Pinch Middle',
            'pinch_ring': 'Pinch Ring',
            'pinch_pinky': 'Pinch Pinky'
        }
        
        for gesture_key, mapping in self.config['mappings'].items():
            display_name = gesture_display_names.get(gesture_key, gesture_key)
            target = mapping['target'].upper()
            mode = mapping['mode'].upper()
            
            # Highlight active gestures
            if gesture_key in self.active_gestures:
                cv2.rectangle(frame, (panel_x - 5, y_offset - 15), (w - 5, y_offset + 3), (0, 255, 0), -1)
                text_color = (0, 0, 0)
            else:
                text_color = (200, 200, 200)
            
            text = f"{display_name}: {target} ({mode})"
            cv2.putText(frame, text, (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            y_offset += 20
        
        # Active gestures in the bottom left
        y_offset = h - 80
        if self.active_gestures:
            cv2.putText(frame, "Active:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            for gesture in self.active_gestures:
                if gesture in self.config['mappings']:
                    mapping = self.config['mappings'][gesture]
                    display_name = gesture_display_names.get(gesture, gesture)
                    text = f"{display_name} -> {mapping['target'].upper()}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'p' to pause", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.controller.release_all()
        try:
            self.hands.close()
        except:
            pass  # Already closed
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hand to Key - Simple Hand Gesture Control"
    )
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Override camera in config if specified
    config = None
    if args.config:
        config = args.config
    
    app = HandToKeyApp(config_file=config)
    
    # Override camera if specified
    if args.camera != 0:
        app.config['camera_index'] = args.camera
        app.cap.release()
        app.cap = cv2.VideoCapture(args.camera)
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
