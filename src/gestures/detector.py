"""
Hand gesture detection module with enhanced features and improved collision handling.

This module provides a sophisticated gesture detection system with the following features:
1. Multiple gesture type detection (open palm, pinch, fist, etc.)
2. Confidence-based gesture scoring
3. Temporal filtering for stability
4. Priority-based collision resolution
5. Enhanced handling of conflicting gestures (e.g., pinch vs. open palm)

The collision detection system uses several strategies to resolve conflicts:
- Gesture priority hierarchy (pinch gestures have higher priority than open palm)
- Confidence-based filtering (higher confidence gestures are preferred)
- Temporal stability (consistent gestures are preferred over transient ones)
- Physical impossibility detection (gestures that cannot occur simultaneously)
- Graduated confidence penalties for borderline cases

This ensures reliable detection even when gestures might be ambiguous or in transition.
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
    
    # Additional gestures for gaming profile
    L_SHAPE = "l_shape"
    HANG_LOOSE = "hang_loose"


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
    """Standard gesture detector with enhanced detection algorithms and improved collision handling."""
    
    # Define gesture priority levels (higher number = higher priority)
    # This helps resolve collisions between conflicting gestures
    GESTURE_PRIORITIES = {
        # Pinch gestures have highest priority to resolve conflicts with open palm
        GestureType.PINCH_INDEX: 12,
        GestureType.PINCH_MIDDLE: 12,
        GestureType.PINCH_RING: 12,
        GestureType.PINCH_PINKY: 12,
        # Advanced gestures have high priority
        GestureType.ROCK_ON: 10,
        # FIST has higher priority than THUMBS_UP to avoid conflicts during transitions
        GestureType.FIST: 9,
        # Gaming profile gestures
        GestureType.L_SHAPE: 8,
        GestureType.HANG_LOOSE: 8,
        # Basic hand shapes have medium priority
        GestureType.VICTORY: 7,
        GestureType.THUMBS_UP: 7,  # Lower priority than fist to resolve conflicts
        GestureType.THREE: 7,
        GestureType.INDEX_ONLY: 7,
        GestureType.OPEN_PALM: 5,
        # Motion gestures have lower priority by default
        GestureType.SWIPE_LEFT: 4,
        GestureType.SWIPE_RIGHT: 4,
        GestureType.SWIPE_UP: 4,
        GestureType.SWIPE_DOWN: 4,
    }
    
    # Define mutually exclusive gesture groups
    # These groups define gestures that cannot physically occur simultaneously
    MUTUALLY_EXCLUSIVE_GROUPS = [
        # Pinch gestures are mutually exclusive with open palm
        {GestureType.OPEN_PALM, GestureType.PINCH_INDEX, GestureType.PINCH_MIDDLE, 
         GestureType.PINCH_RING, GestureType.PINCH_PINKY},
        # Basic hand shapes are mutually exclusive
        {GestureType.OPEN_PALM, GestureType.FIST, GestureType.VICTORY, 
         GestureType.THREE, GestureType.INDEX_ONLY, GestureType.THUMBS_UP,
         GestureType.L_SHAPE, GestureType.HANG_LOOSE},
        # Rock on conflicts with certain gestures
        {GestureType.ROCK_ON, GestureType.OPEN_PALM, GestureType.FIST},
        # L_SHAPE conflicts with certain gestures
        {GestureType.L_SHAPE, GestureType.VICTORY, GestureType.THREE, GestureType.ROCK_ON},
        # HANG_LOOSE conflicts with certain gestures
        {GestureType.HANG_LOOSE, GestureType.ROCK_ON, GestureType.THUMBS_UP},
    ]
    
    # Minimum confidence threshold for gesture detection
    # Gestures below this threshold will be filtered out during conflict resolution
    CONFIDENCE_THRESHOLD = 0.4
    
    def __init__(self, sensitivity: float = 1.0, 
                 enable_motion_gestures: bool = True,
                 enable_custom_gestures: bool = False,
                 enabled_gestures: Optional[Set[str]] = None):
        self.sensitivity = sensitivity
        self.enable_motion_gestures = enable_motion_gestures
        self.enable_custom_gestures = enable_custom_gestures
        # Convert enabled_gestures to GestureType set, or enable all if None
        if enabled_gestures is not None:
            self.enabled_gestures = set()
            for gesture_str in enabled_gestures:
                try:
                    gesture_type = GestureType(gesture_str)
                    self.enabled_gestures.add(gesture_type)
                except ValueError:
                    logger.warning(f"Unknown gesture type in enabled_gestures: {gesture_str}")
        else:
            # If no enabled_gestures specified, enable all gestures
            self.enabled_gestures = set(GestureType)
        
        self.motion_history: List[Tuple[float, float]] = []
        self.custom_gesture_models = {}
        # Store gesture confidence scores
        self.gesture_confidences: Dict[GestureType, float] = {}
        
    def detect(self, landmarks: Any) -> Set[GestureType]:
        """Detect all applicable gestures from hand landmarks with performance optimizations."""
        gestures = set()
        self.gesture_confidences = {}
        
        # Debug: Log enabled gestures occasionally for troubleshooting
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 150 == 0:  # Log every ~5 seconds at 30 FPS
            logger.info(f"Enabled gestures: {[g.value for g in self.enabled_gestures]}")
        
        # Cache expensive finger extension calculations to avoid redundant computation
        self._cached_finger_states = {
            'thumb': self._thumb_extended(landmarks),
            'index': self._finger_extended(landmarks, 8, 6),
            'middle': self._finger_extended(landmarks, 12, 10),
            'ring': self._finger_extended(landmarks, 16, 14),
            'pinky': self._finger_extended(landmarks, 20, 18)
        }
        
        # Detect pinch gestures first (higher priority) - only if enabled
        has_pinch = False
        if GestureType.PINCH_INDEX in self.enabled_gestures and self._is_pinch(landmarks, 8):
            gestures.add(GestureType.PINCH_INDEX)
            has_pinch = True
        if GestureType.PINCH_MIDDLE in self.enabled_gestures and self._is_pinch(landmarks, 12):
            gestures.add(GestureType.PINCH_MIDDLE)
            has_pinch = True
        if GestureType.PINCH_RING in self.enabled_gestures and self._is_pinch(landmarks, 16):
            gestures.add(GestureType.PINCH_RING)
            has_pinch = True
        if GestureType.PINCH_PINKY in self.enabled_gestures and self._is_pinch(landmarks, 20):
            gestures.add(GestureType.PINCH_PINKY)
            has_pinch = True
            
        # Advanced gestures - only if enabled
        if GestureType.ROCK_ON in self.enabled_gestures and self._is_rock_on(landmarks):
            gestures.add(GestureType.ROCK_ON)
        if GestureType.L_SHAPE in self.enabled_gestures and self._is_l_shape(landmarks):
            gestures.add(GestureType.L_SHAPE)
        if GestureType.HANG_LOOSE in self.enabled_gestures and self._is_hang_loose(landmarks):
            gestures.add(GestureType.HANG_LOOSE)
            
        # Basic gesture detection - skip open palm if pinch is detected or if disabled
        if GestureType.OPEN_PALM in self.enabled_gestures and not has_pinch and self._is_open_palm(landmarks):
            gestures.add(GestureType.OPEN_PALM)
        
        # Detect fist first (higher priority than thumbs up) - only if enabled
        has_fist = False
        if GestureType.FIST in self.enabled_gestures and self._is_fist_optimized(landmarks):
            gestures.add(GestureType.FIST)
            has_fist = True
        
        # Other basic gestures - only if enabled
        if GestureType.VICTORY in self.enabled_gestures and self._is_victory(landmarks):
            gestures.add(GestureType.VICTORY)
        if GestureType.THREE in self.enabled_gestures and self._is_three_fingers(landmarks):
            gestures.add(GestureType.THREE)
        if GestureType.INDEX_ONLY in self.enabled_gestures and self._is_index_only(landmarks):
            gestures.add(GestureType.INDEX_ONLY)
        
        # Only check thumbs up if enabled, fist is not detected, and not disabled
        if GestureType.THUMBS_UP in self.enabled_gestures and not has_fist and self._is_thumbs_up_optimized(landmarks):
            gestures.add(GestureType.THUMBS_UP)
            
        # Motion gestures - only check enabled ones
        if self.enable_motion_gestures:
            motion_gesture = self._detect_motion_gesture(landmarks)
            if motion_gesture and motion_gesture in self.enabled_gestures:
                gestures.add(motion_gesture)
        
        # Clear cached finger states after detection
        self._cached_finger_states = None
        
        # Apply mutual exclusion rules based on priorities
        return self._resolve_gesture_conflicts(gestures)
        
    def _resolve_gesture_conflicts(self, detected_gestures: Set[GestureType]) -> Set[GestureType]:
        """
        Resolve conflicts between mutually exclusive gestures based on priorities and confidence scores.
        
        This enhanced conflict resolution system:
        1. Filters out low-confidence gestures
        2. Prioritizes gestures based on their priority level
        3. Uses confidence scores as a tiebreaker when priorities are equal
        4. Handles special case conflicts (like pinch vs. open palm)
        """
        if not detected_gestures:
            return detected_gestures
        
        # First, filter out low-confidence gestures
        filtered_gestures = set()
        for gesture in detected_gestures:
            confidence = self.gesture_confidences.get(gesture, 0.0)
            if confidence >= self.CONFIDENCE_THRESHOLD:
                filtered_gestures.add(gesture)
        
        # If filtering removed all gestures, return the empty set
        if not filtered_gestures:
            return filtered_gestures
        
        # Special case: Pinch gestures always take precedence over open palm
        # This handles the specific collision case mentioned in the requirements
        has_pinch = any(g in {GestureType.PINCH_INDEX, GestureType.PINCH_MIDDLE, 
                             GestureType.PINCH_RING, GestureType.PINCH_PINKY} 
                        for g in filtered_gestures)
        
        if has_pinch and GestureType.OPEN_PALM in filtered_gestures:
            filtered_gestures.remove(GestureType.OPEN_PALM)
            
        # Check for conflicts within each mutually exclusive group
        for group in self.MUTUALLY_EXCLUSIVE_GROUPS:
            intersection = group.intersection(filtered_gestures)
            if len(intersection) > 1:
                # Find gestures with highest priority in this group
                max_priority = max(self.GESTURE_PRIORITIES.get(g, 0) for g in intersection)
                highest_priority_gestures = [g for g in intersection 
                                           if self.GESTURE_PRIORITIES.get(g, 0) == max_priority]
                
                # If multiple gestures have the same priority, use confidence as tiebreaker
                if len(highest_priority_gestures) > 1:
                    highest_confidence_gesture = max(
                        highest_priority_gestures,
                        key=lambda g: self.gesture_confidences.get(g, 0.0)
                    )
                    highest_priority_gestures = [highest_confidence_gesture]
                
                # Remove all other gestures from this group
                for gesture in intersection:
                    if gesture not in highest_priority_gestures:
                        filtered_gestures.remove(gesture)
                        
        return filtered_gestures
    
    def get_confidence(self, gesture_type: GestureType, landmarks: Any) -> float:
        """Calculate confidence score for a specific gesture."""
        # First check if we already calculated confidence during detection
        if gesture_type in self.gesture_confidences:
            return self.gesture_confidences[gesture_type]
        
        # Otherwise use specific confidence calculators
        confidence_map = {
            GestureType.OPEN_PALM: self._get_open_palm_confidence,
            GestureType.FIST: self._get_fist_confidence,
            GestureType.VICTORY: self._get_victory_confidence,
            GestureType.THUMBS_UP: self._get_thumbs_up_confidence,
            # Add more confidence calculators
        }
        
        calculator = confidence_map.get(gesture_type)
        if calculator:
            confidence = calculator(landmarks)
            # Store for future reference
            self.gesture_confidences[gesture_type] = confidence
            return confidence
            
        # Default confidence for gestures without specific calculators
        return 0.8  # Reasonable default confidence
    
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
        """
        Detect open palm gesture with enhanced collision handling.
        
        An open palm is detected when all five fingers (including thumb) are extended
        and there are no active pinch gestures. This implementation includes improved
        confidence scoring to better handle conflicts with pinch gestures.
        """
        # Check if all fingers are extended
        finger_extension = [
            self._thumb_extended(landmarks),          # Thumb
            self._finger_extended(landmarks, 8, 6),   # Index
            self._finger_extended(landmarks, 12, 10), # Middle
            self._finger_extended(landmarks, 16, 14), # Ring
            self._finger_extended(landmarks, 20, 18)  # Pinky
        ]
        all_fingers_extended = all(finger_extension)
        
        # Check for potential pinch gestures
        pinch_distances = [
            self._l2_distance(landmarks[4], landmarks[8]),   # Thumb-Index distance
            self._l2_distance(landmarks[4], landmarks[12]),  # Thumb-Middle distance
            self._l2_distance(landmarks[4], landmarks[16]),  # Thumb-Ring distance
            self._l2_distance(landmarks[4], landmarks[20])   # Thumb-Pinky distance
        ]
        
        # Normalize distances by hand size
        diag = self._bbox_diag(landmarks) + 1e-6
        normalized_distances = [d / diag for d in pinch_distances]
        
        # Check if any pinch gestures are active
        pinch_threshold = 0.08 * self.sensitivity
        pinch_active = any(d < pinch_threshold for d in normalized_distances)
        
        # Calculate confidence score for open palm
        confidence = 0.0
        if all_fingers_extended:
            # Base confidence from finger extension
            confidence = 0.7
            
            # Increase confidence if fingers are well-separated
            if self._fingers_are_spread(landmarks):
                confidence += 0.2
            
            # Calculate pinch-related confidence penalty
            # The closer any finger is to the thumb, the lower the confidence
            min_normalized_distance = min(normalized_distances)
            pinch_ratio = min_normalized_distance / pinch_threshold
            
            # Apply a graduated penalty based on how close we are to a pinch
            if pinch_ratio < 2.0:  # Within 2x the pinch threshold
                # Penalty increases as we get closer to pinch threshold
                penalty = 0.4 * max(0, 1 - (pinch_ratio / 2.0))
                confidence -= penalty
                
        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, confidence))
                
        # Store confidence score for later use
        self.gesture_confidences[GestureType.OPEN_PALM] = confidence
        
        # Only detect open palm if all fingers are extended and no pinch is active
        return all_fingers_extended and not pinch_active
    
    def _is_fist(self, landmarks: List[Any]) -> bool:
        """Detect fist gesture."""
        is_fist = not any([
            self._finger_extended(landmarks, 8, 6),
            self._finger_extended(landmarks, 12, 10),
            self._finger_extended(landmarks, 16, 14),
            self._finger_extended(landmarks, 20, 18),
            self._thumb_extended(landmarks)
        ])
        
        if is_fist:
            # Calculate confidence based on how curled the fingers are
            confidence = self._get_fist_confidence(landmarks)
            self.gesture_confidences[GestureType.FIST] = confidence
        
        return is_fist
    
    def _is_victory(self, landmarks: List[Any]) -> bool:
        """Detect victory/peace gesture."""
        is_victory = (self._finger_extended(landmarks, 8, 6) and 
                     self._finger_extended(landmarks, 12, 10) and
                     not self._finger_extended(landmarks, 16, 14) and
                     not self._finger_extended(landmarks, 20, 18))
        
        if is_victory:
            confidence = self._get_victory_confidence(landmarks)
            self.gesture_confidences[GestureType.VICTORY] = confidence
        
        return is_victory
    
    def _is_three_fingers(self, landmarks: List[Any]) -> bool:
        """Detect three fingers gesture."""
        is_three = (self._finger_extended(landmarks, 8, 6) and 
                   self._finger_extended(landmarks, 12, 10) and
                   self._finger_extended(landmarks, 16, 14) and
                   not self._finger_extended(landmarks, 20, 18))
        
        if is_three:
            confidence = self._get_three_fingers_confidence(landmarks)
            self.gesture_confidences[GestureType.THREE] = confidence
        
        return is_three
    
    def _is_index_only(self, landmarks: List[Any]) -> bool:
        """Detect index finger only gesture."""
        is_index_only = (self._finger_extended(landmarks, 8, 6) and
                        not self._finger_extended(landmarks, 12, 10) and
                        not self._finger_extended(landmarks, 16, 14) and
                        not self._finger_extended(landmarks, 20, 18) and
                        not self._thumb_extended(landmarks))
        
        if is_index_only:
            confidence = self._get_index_only_confidence(landmarks)
            self.gesture_confidences[GestureType.INDEX_ONLY] = confidence
        
        return is_index_only
    
    def _is_thumbs_up(self, landmarks: List[Any]) -> bool:
        """Detect thumbs up gesture."""
        is_thumbs_up = (self._thumb_extended(landmarks) and
                       not self._finger_extended(landmarks, 8, 6) and
                       not self._finger_extended(landmarks, 12, 10) and
                       not self._finger_extended(landmarks, 16, 14) and
                       not self._finger_extended(landmarks, 20, 18))
        
        if is_thumbs_up:
            confidence = self._get_thumbs_up_confidence(landmarks)
            self.gesture_confidences[GestureType.THUMBS_UP] = confidence
        
        return is_thumbs_up
    
    def _fingers_are_spread(self, landmarks: List[Any]) -> bool:
        """Check if fingers are well-separated from each other."""
        # Get fingertip positions
        fingertips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        # Calculate minimum distance between any two fingertips
        min_distance = float('inf')
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = self._l2_distance(fingertips[i], fingertips[j])
                min_distance = min(min_distance, dist)
        
        # Normalize by hand size
        diag = self._bbox_diag(landmarks) + 1e-6
        spread_ratio = min_distance / diag
        
        # Fingers are considered spread if the minimum distance between
        # any two fingertips is at least 10% of the hand's bounding box diagonal
        return spread_ratio > 0.1
    
    def _is_pinch(self, landmarks: List[Any], tip_idx: int) -> bool:
        """Detect pinch gesture with specific finger."""
        ratio = 0.08 * self.sensitivity
        d = self._l2_distance(landmarks[4], landmarks[tip_idx])
        diag = self._bbox_diag(landmarks) + 1e-6
        
        # Calculate confidence based on distance
        pinch_ratio = d / (ratio * diag)
        confidence = max(0.0, 1.0 - pinch_ratio)
        
        # Store confidence score for this pinch gesture
        if tip_idx == 8:
            self.gesture_confidences[GestureType.PINCH_INDEX] = confidence
        elif tip_idx == 12:
            self.gesture_confidences[GestureType.PINCH_MIDDLE] = confidence
        elif tip_idx == 16:
            self.gesture_confidences[GestureType.PINCH_RING] = confidence
        elif tip_idx == 20:
            self.gesture_confidences[GestureType.PINCH_PINKY] = confidence
        
        return d < ratio * diag
    
    def _is_rock_on(self, landmarks: List[Any]) -> bool:
        """Detect rock on gesture (index and pinky extended)."""
        is_rock_on = (self._finger_extended(landmarks, 8, 6) and
                     not self._finger_extended(landmarks, 12, 10) and
                     not self._finger_extended(landmarks, 16, 14) and
                     self._finger_extended(landmarks, 20, 18))
        
        if is_rock_on:
            # Calculate confidence based on finger states
            index_ext = float(self._finger_extended(landmarks, 8, 6))
            middle_curl = float(not self._finger_extended(landmarks, 12, 10))
            ring_curl = float(not self._finger_extended(landmarks, 16, 14))
            pinky_ext = float(self._finger_extended(landmarks, 20, 18))
            
            confidence = (index_ext + middle_curl + ring_curl + pinky_ext) / 4.0
            self.gesture_confidences[GestureType.ROCK_ON] = confidence
        
        return is_rock_on
                
    def _is_l_shape(self, landmarks: List[Any]) -> bool:
        """Detect L shape gesture (thumb and index extended at right angle)."""
        is_l_shape = (self._thumb_extended(landmarks) and
                     self._finger_extended(landmarks, 8, 6) and
                     not self._finger_extended(landmarks, 12, 10) and
                     not self._finger_extended(landmarks, 16, 14) and
                     not self._finger_extended(landmarks, 20, 18))
        
        if is_l_shape:
            # Calculate confidence based on finger states
            thumb_ext = float(self._thumb_extended(landmarks))
            index_ext = float(self._finger_extended(landmarks, 8, 6))
            middle_curl = float(not self._finger_extended(landmarks, 12, 10))
            ring_curl = float(not self._finger_extended(landmarks, 16, 14))
            pinky_curl = float(not self._finger_extended(landmarks, 20, 18))
            
            confidence = (thumb_ext * 0.25 + index_ext * 0.25 + middle_curl * 0.17 + ring_curl * 0.16 + pinky_curl * 0.17)
            self.gesture_confidences[GestureType.L_SHAPE] = confidence
        
        return is_l_shape
                
    def _is_hang_loose(self, landmarks: List[Any]) -> bool:
        """Detect hang loose gesture (thumb and pinky extended, middle fingers closed)."""
        is_hang_loose = (self._thumb_extended(landmarks) and
                        not self._finger_extended(landmarks, 8, 6) and
                        not self._finger_extended(landmarks, 12, 10) and
                        not self._finger_extended(landmarks, 16, 14) and
                        self._finger_extended(landmarks, 20, 18))
        
        if is_hang_loose:
            # Calculate confidence based on finger states
            thumb_ext = float(self._thumb_extended(landmarks))
            index_curl = float(not self._finger_extended(landmarks, 8, 6))
            middle_curl = float(not self._finger_extended(landmarks, 12, 10))
            ring_curl = float(not self._finger_extended(landmarks, 16, 14))
            pinky_ext = float(self._finger_extended(landmarks, 20, 18))
            
            confidence = (thumb_ext * 0.25 + index_curl * 0.19 + middle_curl * 0.18 + ring_curl * 0.18 + pinky_ext * 0.20)
            self.gesture_confidences[GestureType.HANG_LOOSE] = confidence
        
        return is_hang_loose
    
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
    
    def _get_three_fingers_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for three fingers gesture."""
        index_extended = self._finger_extended(landmarks, 8, 6)
        middle_extended = self._finger_extended(landmarks, 12, 10)
        ring_extended = self._finger_extended(landmarks, 16, 14)
        pinky_curled = not self._finger_extended(landmarks, 20, 18)
        
        score = 0.0
        if index_extended: score += 0.25
        if middle_extended: score += 0.25
        if ring_extended: score += 0.25
        if pinky_curled: score += 0.25
        
        return score
    
    def _get_index_only_confidence(self, landmarks: List[Any]) -> float:
        """Calculate confidence for index only gesture."""
        index_extended = self._finger_extended(landmarks, 8, 6)
        middle_curled = not self._finger_extended(landmarks, 12, 10)
        ring_curled = not self._finger_extended(landmarks, 16, 14)
        pinky_curled = not self._finger_extended(landmarks, 20, 18)
        thumb_curled = not self._thumb_extended(landmarks)
        
        score = 0.0
        if index_extended: score += 0.3
        if middle_curled: score += 0.175
        if ring_curled: score += 0.175
        if pinky_curled: score += 0.175
        if thumb_curled: score += 0.175
        
        return score
    
    def _is_fist_optimized(self, landmarks: List[Any]) -> bool:
        """Optimized fist detection using cached finger states to avoid redundant calculations."""
        # Use cached finger states if available
        if hasattr(self, '_cached_finger_states') and self._cached_finger_states is not None:
            is_fist = not any([
                self._cached_finger_states['index'],
                self._cached_finger_states['middle'],
                self._cached_finger_states['ring'],
                self._cached_finger_states['pinky'],
                self._cached_finger_states['thumb']
            ])
        else:
            # Fallback to regular detection if caching is not available
            is_fist = not any([
                self._finger_extended(landmarks, 8, 6),
                self._finger_extended(landmarks, 12, 10),
                self._finger_extended(landmarks, 16, 14),
                self._finger_extended(landmarks, 20, 18),
                self._thumb_extended(landmarks)
            ])
        
        if is_fist:
            # Calculate confidence with enhanced scoring for fist vs thumbs up disambiguation
            confidence = self._get_fist_confidence_optimized(landmarks)
            self.gesture_confidences[GestureType.FIST] = confidence
        
        return is_fist
    
    def _is_thumbs_up_optimized(self, landmarks: List[Any]) -> bool:
        """Optimized thumbs up detection with enhanced requirements to avoid false positives and conflicts with fist."""
        # Use cached finger states if available for better performance
        if hasattr(self, '_cached_finger_states') and self._cached_finger_states is not None:
            # Basic thumbs up detection: thumb extended, other fingers curled
            is_thumbs_up_basic = (
                self._cached_finger_states['thumb'] and
                not self._cached_finger_states['index'] and
                not self._cached_finger_states['middle'] and
                not self._cached_finger_states['ring'] and
                not self._cached_finger_states['pinky']
            )
        else:
            # Fallback to regular detection if caching is not available
            is_thumbs_up_basic = (
                self._thumb_extended(landmarks) and
                not self._finger_extended(landmarks, 8, 6) and
                not self._finger_extended(landmarks, 12, 10) and
                not self._finger_extended(landmarks, 16, 14) and
                not self._finger_extended(landmarks, 20, 18)
            )
        
        if not is_thumbs_up_basic:
            return False
        
        # Enhanced requirements for thumbs up to avoid false positives during fist transitions
        # Check thumb position relative to other fingers - thumb should be clearly extended upward
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        wrist = landmarks[0]
        
        # Thumb should be significantly higher than wrist and thumb base
        thumb_vertical_extension = abs(thumb_tip.y - wrist.y) / abs(thumb_base.y - wrist.y + 1e-6)
        
        # Check that fingers are well-curled (not just barely curled)
        finger_curl_scores = []
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            # Finger should be curled significantly
            curl_ratio = self._l2_distance(tip, wrist) / (self._l2_distance(pip, wrist) + 1e-6)
            finger_curl_scores.append(1.0 - min(1.0, max(0.0, curl_ratio - 0.8) / 0.4))  # Normalize curl score
        
        avg_curl_score = np.mean(finger_curl_scores)
        
        # Stricter requirements: thumb must be clearly extended and fingers clearly curled
        is_clear_thumbs_up = (
            thumb_vertical_extension > 1.2 and  # Thumb clearly extended relative to base
            avg_curl_score > 0.7  # Fingers clearly curled
        )
        
        if is_clear_thumbs_up:
            # Calculate confidence with stricter scoring
            confidence = self._get_thumbs_up_confidence_optimized(landmarks, thumb_vertical_extension, avg_curl_score)
            self.gesture_confidences[GestureType.THUMBS_UP] = confidence
        
        return is_clear_thumbs_up
    
    def _get_fist_confidence_optimized(self, landmarks: List[Any]) -> float:
        """Optimized confidence calculation for fist gesture with enhanced scoring vs thumbs up."""
        # Use cached states for efficiency if available
        if hasattr(self, '_cached_finger_states') and self._cached_finger_states is not None:
            # All fingers and thumb should be curled
            curl_score = sum([
                not self._cached_finger_states['index'],
                not self._cached_finger_states['middle'],
                not self._cached_finger_states['ring'],
                not self._cached_finger_states['pinky'],
                not self._cached_finger_states['thumb']
            ]) / 5.0
        else:
            # Fallback calculation
            curl_score = sum([
                not self._finger_extended(landmarks, 8, 6),
                not self._finger_extended(landmarks, 12, 10),
                not self._finger_extended(landmarks, 16, 14),
                not self._finger_extended(landmarks, 20, 18),
                not self._thumb_extended(landmarks)
            ]) / 5.0
        
        # Enhanced confidence calculation
        finger_scores = []
        wrist = landmarks[0]
        
        # Check how curled each finger actually is
        for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            curl_ratio = self._l2_distance(tip, wrist) / (self._l2_distance(pip, wrist) + 1e-6)
            finger_scores.append(min(1.0, (1.2 - curl_ratio) / 0.3))
        
        # Include thumb curl score
        thumb_tip = landmarks[4]
        thumb_pip = landmarks[3]
        thumb_curl_ratio = self._l2_distance(thumb_tip, wrist) / (self._l2_distance(thumb_pip, wrist) + 1e-6)
        finger_scores.append(min(1.0, (1.2 - thumb_curl_ratio) / 0.3))
        
        # Combine basic curl score with detailed finger analysis
        detailed_score = np.mean(finger_scores)
        final_confidence = (curl_score * 0.4 + detailed_score * 0.6)
        
        # Boost confidence if this is clearly a fist (all fingers well-curled)
        if curl_score >= 1.0 and detailed_score > 0.8:
            final_confidence = min(1.0, final_confidence + 0.1)
        
        return final_confidence
    
    def _get_thumbs_up_confidence_optimized(self, landmarks: List[Any], 
                                           thumb_extension: float, finger_curl: float) -> float:
        """Optimized confidence calculation for thumbs up with stricter requirements."""
        # Base confidence from finger states
        if hasattr(self, '_cached_finger_states') and self._cached_finger_states is not None:
            thumb_extended = self._cached_finger_states['thumb']
            fingers_curled_score = sum([
                not self._cached_finger_states['index'],
                not self._cached_finger_states['middle'],
                not self._cached_finger_states['ring'],
                not self._cached_finger_states['pinky']
            ]) / 4.0
        else:
            # Fallback calculation
            thumb_extended = self._thumb_extended(landmarks)
            fingers_curled_score = sum([
                not self._finger_extended(landmarks, 8, 6),
                not self._finger_extended(landmarks, 12, 10),
                not self._finger_extended(landmarks, 16, 14),
                not self._finger_extended(landmarks, 20, 18)
            ]) / 4.0
        
        # Enhanced confidence based on stricter criteria
        base_confidence = (float(thumb_extended) * 0.3 + fingers_curled_score * 0.3)
        
        # Add bonuses for clear thumb extension and finger curling
        extension_bonus = min(0.2, (thumb_extension - 1.2) * 0.5)  # Bonus for clear thumb extension
        curl_bonus = min(0.2, (finger_curl - 0.7) * 0.67)  # Bonus for clear finger curling
        
        final_confidence = base_confidence + extension_bonus + curl_bonus
        
        # Penalty if the gesture is ambiguous (could be transitioning to/from fist)
        if thumb_extension < 1.4 or finger_curl < 0.8:
            final_confidence *= 0.9  # Small penalty for ambiguous cases
        
        return min(1.0, max(0.0, final_confidence))


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
                 confidence_threshold: float = 0.6,
                 temporal_smoothing: bool = True,
                 smoothing_window: int = 3,
                 min_gesture_frames: int = 2,
                 hand_preference: str = "right",
                 config_manager=None):
        
        self.detector = detector
        self.enable_multi_hand = enable_multi_hand
        self.max_hands = max_hands if enable_multi_hand else 1
        self.enable_gpu = enable_gpu
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.min_gesture_frames = min_gesture_frames
        self.hand_preference = hand_preference
        self.config_manager = config_manager
        
        # Update detector with enabled gestures from config if available
        if self.config_manager and hasattr(self.detector, 'enabled_gestures'):
            self.update_enabled_gestures_from_config()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Processing state
        self.hand_trackings: Dict[int, HandTracking] = {}
        self.frame_counter = 0
        self.processing_time_avg = 0.0
        
        # Enhanced hand tracking for persistence
        self.persistent_hand_tracking: Optional[HandTracking] = None
        self.hand_lost_frames = 0
        self.max_lost_frames = 10  # Allow up to 10 frames without detection before clearing
        self.redetection_threshold = 0.3  # Distance threshold for hand redetection
        
        # Gesture history for temporal filtering
        self.gesture_history: Dict[int, List[Set[GestureType]]] = {}
        self.last_reported_gestures: Dict[int, Set[GestureType]] = {}
        self.gesture_stability_count: Dict[Tuple[int, GestureType], int] = {}
        
        # Optimized threading with dynamic queue sizes
        # Adjust queue sizes based on performance characteristics
        queue_size = 10 if enable_gpu else 5  # GPU can handle larger queues
        self.processing_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self.processing_thread = None
        self.is_running = False
        
        # Performance monitoring for dynamic adjustments
        self.performance_samples = []
        self.max_performance_samples = 100
        self.queue_adjustment_counter = 0
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'gesture_stability': 0.0
        }
        
        # Initialize MediaPipe Hands with error handling and performance optimization
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                model_complexity=0,  # Use model complexity 0 for better performance
                min_detection_confidence=max(0.5, confidence_threshold * 0.8),  # Lower detection threshold for redetection
                min_tracking_confidence=max(0.5, confidence_threshold * 0.8)    # Slightly higher tracking threshold for stability
            )
            self.mediapipe_initialized = True
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Hands: {e}")
            logger.warning("Falling back to basic gesture detection without MediaPipe")
            self.mediapipe_initialized = False
    
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
        """Process a single frame and return detected gestures with optimized performance."""
        self.frame_counter += 1
        
        # Frame skipping for performance - use more aggressive skipping if processing is slow
        skip_factor = self.frame_skip + 1
        if self.stats['avg_processing_time'] > 0.05:  # If processing is slow (>50ms)
            skip_factor += 1  # Skip more frames
        
        if self.frame_skip > 0 and self.frame_counter % skip_factor != 0:
            return []
        
        # Raw detections before temporal filtering
        raw_detections = []
        
        # Map to store detected gestures by hand index
        current_gestures_by_hand: Dict[int, Set[GestureType]] = {}
        
        # Check if MediaPipe is initialized
        if not hasattr(self, 'mediapipe_initialized') or not self.mediapipe_initialized:
            logger.warning("MediaPipe not initialized, returning empty detections")
            return []
        
        try:
            # Optimize frame preprocessing - resize if frame is too large
            height, width = frame.shape[:2]
            if width > 1280 or height > 720:
                # Downsample large frames for better performance
                scale_factor = min(1280 / width, 720 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB with optimized color space conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure frame is contiguous for better MediaPipe performance
            if not rgb_frame.flags['C_CONTIGUOUS']:
                rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Process with MediaPipe
            start_time = time.time()
            results = self.hands.process(rgb_frame)
            processing_time = time.time() - start_time
            
            # Update statistics with exponential smoothing
            self.stats['frames_processed'] += 1
            self.stats['avg_processing_time'] = (
                0.85 * self.stats['avg_processing_time'] + 0.15 * processing_time
            )
            
            # Handle hand redetection and persistence
            self._handle_hand_persistence(results)
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Apply hand preference filtering
                    if not self._should_process_hand(hand_idx, results.multi_handedness):
                        continue
                        
                    # Calculate hand position for persistence tracking
                    palm_center = hand_landmarks.landmark[9]
                    current_position = (palm_center.x, palm_center.y)
                    
                    # Handle hand redetection - try to match with persistent tracking
                    persistent_hand_idx = self._match_persistent_hand(current_position, hand_idx)
                    
                    # Detect gestures
                    gesture_types = self.detector.detect(hand_landmarks.landmark)
                    
                    # Store current gestures for this hand (using persistent index)
                    current_gestures_by_hand[persistent_hand_idx] = gesture_types
                    
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
                            
                            raw_detections.append(detection)
                            
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
        except Exception as e:
            logger.error(f"Error processing frame with MediaPipe: {e}")
            logger.warning("Returning empty detections due to processing error")
            return []
        
        # Apply temporal filtering if enabled
        if self.temporal_smoothing:
            filtered_detections = self._apply_temporal_filtering(raw_detections, current_gestures_by_hand)
            return filtered_detections
        else:
            return raw_detections
            
    def _apply_temporal_filtering(self, 
                                 raw_detections: List[GestureDetection],
                                 current_gestures_by_hand: Dict[int, Set[GestureType]]) -> List[GestureDetection]:
        """
        Apply temporal filtering to stabilize gesture detection with enhanced collision handling.
        
        This improved temporal filtering:
        1. Considers gesture priorities when determining stability
        2. Requires higher stability for lower-priority gestures to override higher-priority ones
        3. Handles special case conflicts between pinch and open palm gestures
        4. Adjusts stability thresholds based on gesture confidence
        """
        filtered_detections = []
        
        # Update gesture history for each hand
        for hand_idx, gestures in current_gestures_by_hand.items():
            # Initialize history for new hands
            if hand_idx not in self.gesture_history:
                self.gesture_history[hand_idx] = []
                self.last_reported_gestures[hand_idx] = set()
            
            # Add current gestures to history
            self.gesture_history[hand_idx].append(gestures)
            
            # Keep history limited to window size
            if len(self.gesture_history[hand_idx]) > self.smoothing_window:
                self.gesture_history[hand_idx].pop(0)
            
            # Update stability counts for each gesture
            for gesture in gestures:
                key = (hand_idx, gesture)
                if key not in self.gesture_stability_count:
                    self.gesture_stability_count[key] = 0
                
                # Get confidence for this gesture
                confidence = 0.0
                for detection in raw_detections:
                    if detection.hand_index == hand_idx and detection.gesture_type == gesture:
                        confidence = detection.confidence
                        break
                
                # Adjust stability increment based on confidence and priority
                # Check if detector has GESTURE_PRIORITIES attribute
                if hasattr(self.detector, 'GESTURE_PRIORITIES'):
                    priority = self.detector.GESTURE_PRIORITIES.get(gesture, 0)
                else:
                    # Fallback priorities for basic gestures
                    priority_map = {
                        GestureType.PINCH_INDEX: 10,
                        GestureType.PINCH_MIDDLE: 10,
                        GestureType.PINCH_RING: 10,
                        GestureType.PINCH_PINKY: 10,
                        GestureType.OPEN_PALM: 5,
                    }
                    priority = priority_map.get(gesture, 0)
                
                # Higher confidence and priority gestures gain stability faster
                increment = 1.0
                if confidence > 0.8:  # High confidence
                    increment += 0.5
                if priority >= 10:     # High priority (pinch gestures)
                    increment += 0.5
                
                self.gesture_stability_count[key] += increment
            
            # Decrease counts for gestures not in current frame
            for key in list(self.gesture_stability_count.keys()):
                h_idx, gesture = key
                if h_idx == hand_idx and gesture not in gestures:
                    # Higher priority gestures decay slower
                    # Check if detector has GESTURE_PRIORITIES attribute
                    if hasattr(self.detector, 'GESTURE_PRIORITIES'):
                        priority = self.detector.GESTURE_PRIORITIES.get(gesture, 0)
                    else:
                        # Fallback priorities for basic gestures
                        priority_map = {
                            GestureType.PINCH_INDEX: 10,
                            GestureType.PINCH_MIDDLE: 10,
                            GestureType.PINCH_RING: 10,
                            GestureType.PINCH_PINKY: 10,
                            GestureType.OPEN_PALM: 5,
                        }
                        priority = priority_map.get(gesture, 0)
                    
                    decay_rate = 1.0
                    if priority >= 10:  # High priority (pinch gestures)
                        decay_rate = 0.7  # Slower decay for high priority gestures
                    
                    self.gesture_stability_count[key] = max(0, self.gesture_stability_count[key] - decay_rate)
                    
                    # Remove if count reaches zero
                    if self.gesture_stability_count[key] == 0:
                        del self.gesture_stability_count[key]
        
        # Determine stable gestures with priority-based conflict resolution
        stable_gestures: Dict[int, Set[GestureType]] = {}
        for hand_idx in current_gestures_by_hand.keys():
            stable_gestures[hand_idx] = set()
            
            # Get all potentially stable gestures for this hand
            potential_stable_gestures = []
            for key, count in self.gesture_stability_count.items():
                h_idx, gesture = key
                if h_idx == hand_idx:
                    # Adjust stability threshold based on priority
                    # Check if detector has GESTURE_PRIORITIES attribute
                    if hasattr(self.detector, 'GESTURE_PRIORITIES'):
                        priority = self.detector.GESTURE_PRIORITIES.get(gesture, 0)
                    else:
                        # Fallback priorities for basic gestures
                        priority_map = {
                            GestureType.PINCH_INDEX: 10,
                            GestureType.PINCH_MIDDLE: 10,
                            GestureType.PINCH_RING: 10,
                            GestureType.PINCH_PINKY: 10,
                            GestureType.OPEN_PALM: 5,
                        }
                        priority = priority_map.get(gesture, 0)
                    
                    # Higher priority gestures need fewer frames to be considered stable
                    threshold_adjustment = 0
                    if priority >= 10:  # High priority (pinch gestures)
                        threshold_adjustment = -1  # Require one fewer frame
                    
                    adjusted_threshold = max(1, self.min_gesture_frames + threshold_adjustment)
                    
                    if count >= adjusted_threshold:
                        potential_stable_gestures.append(gesture)
            
            # Resolve conflicts between stable gestures using detector's conflict resolution
            if potential_stable_gestures:
                # Check if detector has _resolve_gesture_conflicts method
                if hasattr(self.detector, '_resolve_gesture_conflicts'):
                    stable_gestures[hand_idx] = self.detector._resolve_gesture_conflicts(set(potential_stable_gestures))
                else:
                    # Fallback to simple priority-based resolution
                    potential_set = set(potential_stable_gestures)
                    
                    # Check for pinch vs open palm conflict
                    has_pinch = any(g in {GestureType.PINCH_INDEX, GestureType.PINCH_MIDDLE, 
                                         GestureType.PINCH_RING, GestureType.PINCH_PINKY} 
                                    for g in potential_set)
                    
                    if has_pinch and GestureType.OPEN_PALM in potential_set:
                        potential_set.remove(GestureType.OPEN_PALM)
                    
                    stable_gestures[hand_idx] = potential_set
        
        # Filter raw detections based on stable gestures
        for detection in raw_detections:
            hand_idx = detection.hand_index
            gesture = detection.gesture_type
            
            if hand_idx in stable_gestures and gesture in stable_gestures[hand_idx]:
                filtered_detections.append(detection)
                
                # Update last reported gestures
                if hand_idx in self.last_reported_gestures:
                    self.last_reported_gestures[hand_idx].add(gesture)
        
        # Calculate stability metric
        total_stability = 0
        total_gestures = 0
        for hand_idx, gestures in stable_gestures.items():
            for gesture in gestures:
                key = (hand_idx, gesture)
                if key in self.gesture_stability_count:
                    total_stability += min(1.0, self.gesture_stability_count[key] / self.smoothing_window)
                    total_gestures += 1
        
        if total_gestures > 0:
            self.stats['gesture_stability'] = total_stability / total_gestures
        
        return filtered_detections
    
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
    
    def draw_debug_info(self, frame: np.ndarray, detections: List[GestureDetection], 
                       show_landmarks: bool = True, show_connections: bool = True):
        """Draw debug information on frame with customizable visualization options."""
        h, w = frame.shape[:2]
        
        # Debug info: show if we have any detections
        logger.debug(f"Drawing debug info for {len(detections)} detections, landmarks={show_landmarks}, connections={show_connections}")
        
        for detection in detections:
            logger.debug(f"Processing detection: {detection.gesture_type.value}, has_landmarks={detection.landmarks is not None}")
            
            if detection.landmarks:
                # Draw hand landmarks and connections based on settings
                if show_landmarks or show_connections:
                    try:
                        # Custom landmark and connection styles
                        landmark_style = self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0),  # Green landmarks
                            thickness=2,
                            circle_radius=4
                        )
                        
                        connection_style = self.mp_drawing.DrawingSpec(
                            color=(255, 255, 255),  # White connections
                            thickness=2
                        )
                        
                        # Draw based on user preferences
                        if show_landmarks and show_connections:
                            logger.debug("Drawing both landmarks and connections")
                            self.mp_drawing.draw_landmarks(
                                frame, 
                                detection.landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                landmark_style,
                                connection_style
                            )
                        elif show_landmarks:
                            logger.debug("Drawing landmarks only")
                            self.mp_drawing.draw_landmarks(
                                frame, 
                                detection.landmarks,
                                None,  # No connections
                                landmark_style,
                                None
                            )
                        elif show_connections:
                            logger.debug("Drawing connections only")
                            # For connections only, we still need some landmark style
                            invisible_landmark_style = self.mp_drawing.DrawingSpec(
                                color=(0, 0, 0),  # Invisible
                                thickness=0,
                                circle_radius=0
                            )
                            self.mp_drawing.draw_landmarks(
                                frame, 
                                detection.landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                invisible_landmark_style,  # Invisible landmarks
                                connection_style
                            )
                    except Exception as e:
                        logger.error(f"Error drawing landmarks: {e}")
                
                # Draw gesture info
                x = int(detection.position[0] * w)
                y = int(detection.position[1] * h)
                
                # Hand preference indicator
                hand_label = f"Hand {detection.hand_index}"
                if self.hand_preference != "both":
                    hand_label += f" ({self.hand_preference})" 
                
                # Main gesture text
                gesture_text = f"{detection.gesture_type.value}: {detection.confidence:.2f}"
                cv2.putText(frame, gesture_text, (x-70, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Hand preference info
                cv2.putText(frame, hand_label, (x-50, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw velocity vector
                if abs(detection.velocity[0]) > 0.01 or abs(detection.velocity[1]) > 0.01:
                    vx = int(detection.velocity[0] * 100)
                    vy = int(detection.velocity[1] * 100)
                    cv2.arrowedLine(frame, (x, y), (x + vx, y + vy), 
                                   (255, 0, 0), 2)
                    
                    # Velocity magnitude text
                    velocity_mag = np.sqrt(detection.velocity[0]**2 + detection.velocity[1]**2)
                    if velocity_mag > 0.05:  # Only show if significant movement
                        cv2.putText(frame, f"v: {velocity_mag:.2f}", (x + vx + 10, y + vy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Draw center point for hand
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red center point
        
        # Draw statistics
        stats_text = [
            f"FPS: {1.0/self.stats['avg_processing_time']:.1f}" if self.stats['avg_processing_time'] > 0 else "FPS: --",
            f"Confidence: {self.stats['avg_confidence']:.2f}",
            f"Gestures: {self.stats['gestures_detected']}",
            f"Hand Preference: {self.hand_preference.title()}"
        ]
        
        # Background for stats
        cv2.rectangle(frame, (5, 5), (250, 30 + len(stats_text) * 25), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (250, 30 + len(stats_text) * 25), (255, 255, 255), 1)
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def _should_process_hand(self, hand_idx: int, multi_handedness: Any) -> bool:
        """
        Determine whether to process a hand based on the hand preference setting.
        
        Args:
            hand_idx: Index of the hand (0 or 1)
            multi_handedness: MediaPipe handedness classification results
            
        Returns:
            True if the hand should be processed, False otherwise
        """
        # If preference is 'both', process all hands
        if self.hand_preference == "both":
            return True
        
        # If no handedness information available, process all hands as fallback
        if not multi_handedness or hand_idx >= len(multi_handedness):
            logger.warning(f"No handedness information available for hand {hand_idx}, processing anyway")
            return True
        
        # Get the handedness classification for this hand
        try:
            hand_classification = multi_handedness[hand_idx]
            # MediaPipe returns handedness from the perspective of the person in the image
            # So 'Left' means the person's left hand, which appears on the right side of the image
            handedness_label = hand_classification.classification[0].label.lower()
            
            # Map MediaPipe handedness to our preference system
            # Note: MediaPipe's coordinate system is mirrored, so we need to invert
            if handedness_label == "left":
                detected_hand = "left"  # Person's left hand
            elif handedness_label == "right":
                detected_hand = "right"  # Person's right hand
            else:
                # Unknown handedness, process as fallback
                logger.warning(f"Unknown handedness classification: {handedness_label}")
                return True
            
            # Check if this hand matches our preference
            should_process = (detected_hand == self.hand_preference)
            
            logger.debug(f"Hand {hand_idx}: detected as {detected_hand}, preference is {self.hand_preference}, processing: {should_process}")
            return should_process
            
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error accessing handedness classification: {e}, processing hand anyway")
            return True
    
    def _handle_hand_persistence(self, results):
        """Handle hand tracking persistence and redetection."""
        if results.multi_hand_landmarks:
            # Hand detected, reset lost frames counter
            self.hand_lost_frames = 0
            
            # If we only have one hand detected and it's the first detection,
            # or if we need to update persistent tracking
            if len(results.multi_hand_landmarks) == 1:
                palm_center = results.multi_hand_landmarks[0].landmark[9]
                current_position = (palm_center.x, palm_center.y)
                
                # Update or create persistent tracking
                if self.persistent_hand_tracking is None:
                    # New hand detected
                    self.persistent_hand_tracking = HandTracking(hand_id=0)
                    self.persistent_hand_tracking.positions.append(current_position)
                    self.persistent_hand_tracking.last_update = time.time()
                    logger.debug("Started persistent hand tracking")
                else:
                    # Update existing tracking
                    self.persistent_hand_tracking.positions.append(current_position)
                    self.persistent_hand_tracking.last_update = time.time()
                    # Keep position history limited
                    if len(self.persistent_hand_tracking.positions) > 100:
                        self.persistent_hand_tracking.positions.pop(0)
        else:
            # No hand detected, increment lost frames counter
            self.hand_lost_frames += 1
            
            # Clear persistent tracking if hand has been lost too long
            if self.hand_lost_frames > self.max_lost_frames:
                if self.persistent_hand_tracking is not None:
                    logger.debug(f"Clearing persistent hand tracking after {self.hand_lost_frames} lost frames")
                    self.persistent_hand_tracking = None
                    self.hand_lost_frames = 0
                    
                    # Clear related tracking data
                    self.hand_trackings.clear()
                    self.gesture_history.clear()
                    self.last_reported_gestures.clear()
                    # Keep some gesture stability for quick redetection
                    for key in list(self.gesture_stability_count.keys()):
                        self.gesture_stability_count[key] *= 0.5  # Reduce stability but don't clear
    
    def _match_persistent_hand(self, current_position: Tuple[float, float], mediapipe_hand_idx: int) -> int:
        """Match a detected hand with persistent tracking."""
        # If no persistent tracking, use MediaPipe index
        if self.persistent_hand_tracking is None:
            return mediapipe_hand_idx
        
        # If we have persistent tracking, try to match based on position
        if self.persistent_hand_tracking.positions:
            last_position = self.persistent_hand_tracking.positions[-1]
            distance = np.sqrt(
                (current_position[0] - last_position[0])**2 + 
                (current_position[1] - last_position[1])**2
            )
            
            # If the hand is close to the last known position, consider it the same hand
            if distance < self.redetection_threshold:
                logger.debug(f"Hand matched with persistent tracking (distance: {distance:.3f})")
                return 0  # Use consistent index 0 for persistent hand
            else:
                logger.debug(f"Hand position changed significantly (distance: {distance:.3f}), treating as new hand")
                # Create new persistent tracking for this position
                self.persistent_hand_tracking = HandTracking(hand_id=0)
                return 0
        
        return mediapipe_hand_idx
    
    def update_enabled_gestures_from_config(self):
        """Update the detector's enabled gestures based on the current config."""
        if not self.config_manager or not hasattr(self.detector, 'enabled_gestures'):
            return
        
        try:
            # Get the active profile from config manager
            active_profile = self.config_manager.active_profile
            if not active_profile:
                logger.warning("No active profile found, keeping all gestures enabled")
                return
            
            # Extract enabled gestures from the profile's gesture mappings (list)
            enabled_gestures = set()
            gesture_mappings = active_profile.gesture_mappings
            
            for mapping in gesture_mappings:
                # Only include enabled gestures
                if mapping.enabled:
                    enabled_gestures.add(mapping.gesture)
            
            # Convert to GestureType set and update detector
            detector_enabled_gestures = set()
            for gesture_str in enabled_gestures:
                try:
                    gesture_type = GestureType(gesture_str)
                    detector_enabled_gestures.add(gesture_type)
                except ValueError:
                    logger.warning(f"Unknown gesture type from config: {gesture_str}")
            
            # Update the detector's enabled gestures
            self.detector.enabled_gestures = detector_enabled_gestures
            
            logger.info(f"Updated gesture detector with {len(detector_enabled_gestures)} enabled gestures from config")
            logger.debug(f"Enabled gestures: {[g.value for g in detector_enabled_gestures]}")
            
        except Exception as e:
            logger.error(f"Error updating enabled gestures from config: {e}")
            # Keep current enabled gestures on error
    
    def update_config_manager(self, config_manager):
        """Update the config manager and refresh enabled gestures."""
        self.config_manager = config_manager
        if hasattr(self.detector, 'enabled_gestures'):
            self.update_enabled_gestures_from_config()
