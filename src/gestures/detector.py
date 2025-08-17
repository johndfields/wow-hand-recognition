"""
Hand gesture detection module with enhanced features.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import mediapipe as mp
import cv2
import time
from abc import ABC, abstractmethod
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    OPEN_PALM = "open_palm"
    FIST = "fist"
    VICTORY = "victory"
    THREE = "three"
    INDEX_ONLY = "index_only"
    THUMBS_UP = "thumbs_up"
    PINCH_INDEX = "pinch_index"
    PINCH_MIDDLE = "pinch_middle"
    PINCH_RING = "pinch_ring"
    PINCH_PINKY = "pinch_pinky"
    CUSTOM = "custom"
    
    # New gestures
    PEACE_SIGN = "peace_sign"
    OK_SIGN = "ok_sign"
    ROCK_ON = "rock_on"
    POINT = "point"
    GRAB = "grab"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"


@dataclass
class GestureDetection:
    """Represents a detected gesture with confidence and metadata."""
    gesture_type: GestureType
    confidence: float
    timestamp: float
    hand_index: int = 0
    landmarks: Optional[Any] = None
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandTracking:
    """Tracks hand movement and gesture history."""
    hand_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    gestures: List[GestureDetection] = field(default_factory=list)
    last_update: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    is_stable: bool = True


class GestureDetectorBase(ABC):
    """Abstract base class for gesture detectors."""
    
    @abstractmethod
    def detect(self, landmarks: Any) -> Set[GestureType]:
        """Detect gestures from hand landmarks."""
        pass
    
    @abstractmethod
    def get_confidence(self, gesture_type: GestureType, landmarks: Any) -> float:
        """Get confidence score for a specific gesture."""
        pass


class StandardGestureDetector(GestureDetectorBase):
    """Standard gesture detector with enhanced detection algorithms."""
    
    def __init__(self, sensitivity: float = 1.0, 
                 enable_motion_gestures: bool = True,
                 enable_custom_gestures: bool = False):
        self.sensitivity = sensitivity
        self.enable_motion_gestures = enable_motion_gestures
        self.enable_custom_gestures = enable_custom_gestures
        self.motion_history: List[Tuple[float, float]] = []
        self.custom_gesture_models = {}
        
    def detect(self, landmarks: Any) -> Set[GestureType]:
        """Detect all applicable gestures from hand landmarks."""
        gestures = set()
        
        # Basic gesture detection
        if self._is_open_palm(landmarks):
            gestures.add(GestureType.OPEN_PALM)
        if self._is_fist(landmarks):
            gestures.add(GestureType.FIST)
        if self._is_victory(landmarks):
            gestures.add(GestureType.VICTORY)
        if self._is_three_fingers(landmarks):
            gestures.add(GestureType.THREE)
        if self._is_index_only(landmarks):
            gestures.add(GestureType.INDEX_ONLY)
        if self._is_thumbs_up(landmarks):
            gestures.add(GestureType.THUMBS_UP)
            
        # Pinch gestures
        if self._is_pinch(landmarks, 8):
            gestures.add(GestureType.PINCH_INDEX)
        if self._is_pinch(landmarks, 12):
            gestures.add(GestureType.PINCH_MIDDLE)
        if self._is_pinch(landmarks, 16):
            gestures.add(GestureType.PINCH_RING)
        if self._is_pinch(landmarks, 20):
            gestures.add(GestureType.PINCH_PINKY)
            
        # Advanced gestures
        if self._is_ok_sign(landmarks):
            gestures.add(GestureType.OK_SIGN)
        if self._is_rock_on(landmarks):
            gestures.add(GestureType.ROCK_ON)
            
        # Motion gestures
        if self.enable_motion_gestures:
            motion_gesture = self._detect_motion_gesture(landmarks)
            if motion_gesture:
                gestures.add(motion_gesture)
                
        return gestures
    
    def get_confidence(self, gesture_type: GestureType, landmarks: Any) -> float:
        """Calculate confidence score for a specific gesture."""
        confidence_map = {
            GestureType.OPEN_PALM: self._get_open_palm_confidence,
            GestureType.FIST: self._get_fist_confidence,
            GestureType.VICTORY: self._get_victory_confidence,
            GestureType.THUMBS_UP: self._get_thumbs_up_confidence,
            # Add more confidence calculators
        }
        
        calculator = confidence_map.get(gesture_type)
        if calculator:
            return calculator(landmarks)
        return 0.0
    
    def _l2_distance(self, p1: Any, p2: Any) -> float:
        """Calculate L2 distance between two landmarks."""
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    
    def _bbox_diag(self, landmarks: List[Any]) -> float:
        """Calculate diagonal of bounding box."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    
    def _finger_extended(self, landmarks: List[Any], tip_idx: int, 
                        pip_idx: int, wrist_idx: int = 0) -> bool:
        """Check if a finger is extended."""
        ratio = 1.15 / self.sensitivity
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        wrist = landmarks[wrist_idx]
        return self._l2_distance(tip, wrist) > self._l2_distance(pip, wrist) * ratio
    
    def _thumb_extended(self, landmarks: List[Any]) -> bool:
        """Check if thumb is extended."""
        ratio = 1.10 / self.sensitivity
        return self._finger_extended(landmarks, 4, 3, 0)
    
    def _is_open_palm(self, landmarks: List[Any]) -> bool:
        """Detect open palm gesture."""
        return all([
            self._finger_extended(landmarks, 8, 6),   # Index
            self._finger_extended(landmarks, 12, 10), # Middle
            self._finger_extended(landmarks, 16, 14), # Ring
            self._finger_extended(landmarks, 20, 18)  # Pinky
        ])
    
    def _is_fist(self, landmarks: List[Any]) -> bool:
        """Detect fist gesture."""
        return not any([
            self._finger_extended(landmarks, 8, 6),
            self._finger_extended(landmarks, 12, 10),
            self._finger_extended(landmarks, 16, 14),
            self._finger_extended(landmarks, 20, 18),
            self._thumb_extended(landmarks)
        ])
    
    def _is_victory(self, landmarks: List[Any]) -> bool:
        """Detect victory/peace gesture."""
        return (self._finger_extended(landmarks, 8, 6) and 
                self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18))
    
    def _is_three_fingers(self, landmarks: List[Any]) -> bool:
        """Detect three fingers gesture."""
        return (self._finger_extended(landmarks, 8, 6) and 
                self._finger_extended(landmarks, 12, 10) and
                self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18))
    
    def _is_index_only(self, landmarks: List[Any]) -> bool:
        """Detect index finger only gesture."""
        return (self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18) and
                not self._thumb_extended(landmarks))
    
    def _is_thumbs_up(self, landmarks: List[Any]) -> bool:
        """Detect thumbs up gesture."""
        return (self._thumb_extended(landmarks) and
                not self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18))
    
    def _is_pinch(self, landmarks: List[Any], tip_idx: int) -> bool:
        """Detect pinch gesture with specific finger."""
        ratio = 0.08 * self.sensitivity
        d = self._l2_distance(landmarks[4], landmarks[tip_idx])
        diag = self._bbox_diag(landmarks) + 1e-6
        return d < ratio * diag
    
    def _is_ok_sign(self, landmarks: List[Any]) -> bool:
        """Detect OK sign gesture."""
        # Thumb and index touching, other fingers extended
        return (self._is_pinch(landmarks, 8) and
                self._finger_extended(landmarks, 12, 10) and
                self._finger_extended(landmarks, 16, 14) and
                self._finger_extended(landmarks, 20, 18))
    
    def _is_rock_on(self, landmarks: List[Any]) -> bool:
        """Detect rock on gesture (index and pinky extended)."""
        return (self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                self._finger_extended(landmarks, 20, 18))
    
    def _detect_motion_gesture(self, landmarks: List[Any]) -> Optional[GestureType]:
        """Detect motion-based gestures like swipes."""
        # Track palm center
        palm_center = landmarks[9]  # Middle of palm
        current_pos = (palm_center.x, palm_center.y)
        
        self.motion_history.append(current_pos)
        if len(self.motion_history) > 30:  # Keep last 30 frames
            self.motion_history.pop(0)
        
        if len(self.motion_history) < 10:
            return None
        
        # Calculate motion vector
        start_pos = self.motion_history[0]
        dx = current_pos[0] - start_pos[0]
        dy = current_pos[1] - start_pos[1]
        
        motion_threshold = 0.2
        if abs(dx) > motion_threshold:
            if dx > 0:
                return GestureType.SWIPE_RIGHT
            else:
                return GestureType.SWIPE_LEFT
        elif abs(dy) > motion_threshold:
            if dy > 0:
                return GestureType.SWIPE_DOWN
            else:
                return GestureType.SWIPE_UP
        
        return None
    
    def _get_open_palm_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for open palm gesture."""
        finger_scores = []
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            wrist = landmarks[0]
            extension_ratio = self._l2_distance(tip, wrist) / (self._l2_distance(pip, wrist) + 1e-6)
            finger_scores.append(min(1.0, (extension_ratio - 1.0) / 0.3))
        return np.mean(finger_scores)
    
    def _get_fist_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for fist gesture."""
        finger_scores = []
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            wrist = landmarks[0]
            curl_ratio = self._l2_distance(tip, wrist) / (self._l2_distance(pip, wrist) + 1e-6)
            finger_scores.append(min(1.0, (1.2 - curl_ratio) / 0.3))
        return np.mean(finger_scores)
    
    def _get_victory_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for victory gesture."""
        index_extended = self._finger_extended(landmarks, 8, 6)
        middle_extended = self._finger_extended(landmarks, 12, 10)
        ring_curled = not self._finger_extended(landmarks, 16, 14)
        pinky_curled = not self._finger_extended(landmarks, 20, 18)
        
        score = 0.0
        if index_extended: score += 0.25
        if middle_extended: score += 0.25
        if ring_curled: score += 0.25
        if pinky_curled: score += 0.25
        
        return score
    
    def _get_thumbs_up_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for thumbs up gesture."""
        thumb_extended = self._thumb_extended(landmarks)
        fingers_curled = sum([
            not self._finger_extended(landmarks, 8, 6),
            not self._finger_extended(landmarks, 12, 10),
            not self._finger_extended(landmarks, 16, 14),
            not self._finger_extended(landmarks, 20, 18)
        ]) / 4.0
        
        return (thumb_extended * 0.5 + fingers_curled * 0.5)


class CustomGestureDetector(GestureDetectorBase):
    """Detector for user-defined custom gestures using machine learning."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.gesture_templates = {}
        if model_path:
            self.load_model(model_path)
    
    def detect(self, landmarks: Any) -> Set[GestureType]:
        """Detect custom gestures using trained model."""
        if not self.model:
            return set()
        
        # Convert landmarks to feature vector
        features = self._extract_features(landmarks)
        
        # Predict gesture
        predictions = self.model.predict(features)
        
        gestures = set()
        for gesture_id, confidence in predictions:
            if confidence > 0.7:
                gestures.add(GestureType.CUSTOM)
        
        return gestures
    
    def get_confidence(self, gesture_type: GestureType, landmarks: Any) -> float:
        """Get confidence for custom gesture."""
        if gesture_type != GestureType.CUSTOM or not self.model:
            return 0.0
        
        features = self._extract_features(landmarks)
        predictions = self.model.predict(features)
        
        return max(predictions.values()) if predictions else 0.0
    
    def train_gesture(self, name: str, samples: List[Any]):
        """Train a new custom gesture from samples."""
        # Extract features from samples
        feature_vectors = [self._extract_features(s) for s in samples]
        
        # Store as template for now (simple nearest neighbor)
        self.gesture_templates[name] = feature_vectors
        
        logger.info(f"Trained custom gesture: {name} with {len(samples)} samples")
    
    def _extract_features(self, landmarks: Any) -> np.ndarray:
        """Extract feature vector from landmarks."""
        features = []
        
        # Relative positions to wrist
        wrist = landmarks[0]
        for lm in landmarks[1:]:
            features.extend([lm.x - wrist.x, lm.y - wrist.y])
        
        # Angles between fingers
        for i in range(1, 5):
            base_idx = i * 4 + 1
            for j in range(3):
                if base_idx + j + 1 < len(landmarks):
                    v1 = np.array([landmarks[base_idx + j].x, landmarks[base_idx + j].y])
                    v2 = np.array([landmarks[base_idx + j + 1].x, landmarks[base_idx + j + 1].y])
                    angle = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
                    features.append(angle)
        
        return np.array(features)
    
    def load_model(self, path: str):
        """Load a pre-trained custom gesture model."""
        # Placeholder for model loading
        logger.info(f"Loading custom gesture model from {path}")
        pass
    
    def save_model(self, path: str):
        """Save the current custom gesture model."""
        # Placeholder for model saving
        logger.info(f"Saving custom gesture model to {path}")
        pass


class GestureProcessor:
    """Main gesture processing engine with threading and optimization."""
    
    def __init__(self, 
                 detector: GestureDetectorBase,
                 enable_multi_hand: bool = True,
                 max_hands: int = 2,
                 enable_gpu: bool = True,
                 frame_skip: int = 0,
                 confidence_threshold: float = 0.6):
        
        self.detector = detector
        self.enable_multi_hand = enable_multi_hand
        self.max_hands = max_hands if enable_multi_hand else 1
        self.enable_gpu = enable_gpu
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Processing state
        self.hand_trackings: Dict[int, HandTracking] = {}
        self.frame_counter = 0
        self.processing_time_avg = 0.0
        
        # Threading
        self.processing_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=1 if not enable_gpu else 2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
    
    def start(self):
        """Start the processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Gesture processor started")
    
    def stop(self):
        """Stop the processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            logger.info("Gesture processor stopped")
    
    def process_frame(self, frame: np.ndarray) -> List[GestureDetection]:
        """Process a single frame and return detected gestures."""
        self.frame_counter += 1
        
        # Frame skipping for performance
        if self.frame_skip > 0 and self.frame_counter % (self.frame_skip + 1) != 0:
            return []
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        start_time = time.time()
        results = self.hands.process(rgb_frame)
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['avg_processing_time'] = (
            0.9 * self.stats['avg_processing_time'] + 0.1 * processing_time
        )
        
        detections = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Detect gestures
                gesture_types = self.detector.detect(hand_landmarks.landmark)
                
                # Create detection objects with confidence scores
                for gesture_type in gesture_types:
                    confidence = self.detector.get_confidence(gesture_type, hand_landmarks.landmark)
                    
                    if confidence >= self.confidence_threshold:
                        # Calculate hand position
                        palm_center = hand_landmarks.landmark[9]
                        position = (palm_center.x, palm_center.y)
                        
                        # Calculate velocity if tracking exists
                        velocity = (0.0, 0.0)
                        if hand_idx in self.hand_trackings:
                            tracking = self.hand_trackings[hand_idx]
                            if tracking.positions:
                                last_pos = tracking.positions[-1]
                                dt = time.time() - tracking.last_update
                                if dt > 0:
                                    velocity = (
                                        (position[0] - last_pos[0]) / dt,
                                        (position[1] - last_pos[1]) / dt
                                    )
                        
                        detection = GestureDetection(
                            gesture_type=gesture_type,
                            confidence=confidence,
                            timestamp=time.time(),
                            hand_index=hand_idx,
                            landmarks=hand_landmarks,
                            position=position,
                            velocity=velocity
                        )
                        
                        detections.append(detection)
                        
                        # Update tracking
                        if hand_idx not in self.hand_trackings:
                            self.hand_trackings[hand_idx] = HandTracking(hand_id=hand_idx)
                        
                        tracking = self.hand_trackings[hand_idx]
                        tracking.positions.append(position)
                        tracking.gestures.append(detection)
                        tracking.last_update = time.time()
                        tracking.velocity = velocity
                        
                        # Keep tracking history limited
                        if len(tracking.positions) > 100:
                            tracking.positions.pop(0)
                        if len(tracking.gestures) > 100:
                            tracking.gestures.pop(0)
                        
                        # Update statistics
                        self.stats['gestures_detected'] += 1
                        self.stats['avg_confidence'] = (
                            0.9 * self.stats['avg_confidence'] + 0.1 * confidence
                        )
        
        return detections
    
    def process_frame_async(self, frame: np.ndarray):
        """Queue frame for asynchronous processing."""
        if self.is_running:
            try:
                self.processing_queue.put_nowait(frame)
            except queue.Full:
                logger.warning("Processing queue full, dropping frame")
    
    def get_results(self) -> Optional[List[GestureDetection]]:
        """Get processed results from the queue."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Background processing thread."""
        while self.is_running:
            try:
                frame = self.processing_queue.get(timeout=0.1)
                detections = self.process_frame(frame)
                
                try:
                    self.result_queue.put_nowait(detections)
                except queue.Full:
                    logger.warning("Result queue full, dropping results")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
    
    def draw_debug_info(self, frame: np.ndarray, detections: List[GestureDetection]):
        """Draw debug information on frame."""
        h, w = frame.shape[:2]
        
        for detection in detections:
            if detection.landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    detection.landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
                
                # Draw gesture info
                x = int(detection.position[0] * w)
                y = int(detection.position[1] * h)
                
                text = f"{detection.gesture_type.value}: {detection.confidence:.2f}"
                cv2.putText(frame, text, (x-50, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw velocity vector
                if abs(detection.velocity[0]) > 0.01 or abs(detection.velocity[1]) > 0.01:
                    vx = int(detection.velocity[0] * 100)
                    vy = int(detection.velocity[1] * 100)
                    cv2.arrowedLine(frame, (x, y), (x + vx, y + vy), 
                                   (255, 0, 0), 2)
        
        # Draw statistics
        stats_text = [
            f"FPS: {1.0/self.stats['avg_processing_time']:.1f}" if self.stats['avg_processing_time'] > 0 else "FPS: --",
            f"Confidence: {self.stats['avg_confidence']:.2f}",
            f"Gestures: {self.stats['gestures_detected']}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
