#!/usr/bin/env python3
"""
Test cases for the gesture detector with focus on collision detection.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path to import our gesture detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gestures.detector import StandardGestureDetector, GestureType

class TestGestureDetector(unittest.TestCase):
    """Test cases for the StandardGestureDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = StandardGestureDetector(sensitivity=1.0)
        
    def test_pinch_open_palm_collision(self):
        """Test that pinch gestures take precedence over open palm when both are detected."""
        # Create mock landmarks that could be interpreted as both pinch and open palm
        landmarks = self._create_mock_landmarks_for_pinch_and_open_palm()
        
        # Detect gestures
        gestures = self.detector.detect(landmarks)
        
        # Verify that pinch is detected and open palm is not
        self.assertIn(GestureType.PINCH_INDEX, gestures)
        self.assertNotIn(GestureType.OPEN_PALM, gestures)
        
        # Verify confidence scores
        pinch_confidence = self.detector.gesture_confidences.get(GestureType.PINCH_INDEX, 0)
        open_palm_confidence = self.detector.gesture_confidences.get(GestureType.OPEN_PALM, 0)
        
        # Pinch should have higher confidence
        self.assertGreater(pinch_confidence, open_palm_confidence)
        
    def test_confidence_based_filtering(self):
        """Test that low-confidence gestures are filtered out during conflict resolution."""
        # Create mock landmarks with ambiguous gesture
        landmarks = self._create_mock_landmarks_with_low_confidence()
        
        # Override confidence scores for testing
        self.detector.detect(landmarks)
        self.detector.gesture_confidences[GestureType.OPEN_PALM] = 0.3  # Below threshold
        self.detector.gesture_confidences[GestureType.FIST] = 0.6  # Above threshold
        
        # Create a set with both gestures
        gestures = {GestureType.OPEN_PALM, GestureType.FIST}
        
        # Resolve conflicts
        resolved_gestures = self.detector._resolve_gesture_conflicts(gestures)
        
        # Verify that only the high-confidence gesture remains
        self.assertNotIn(GestureType.OPEN_PALM, resolved_gestures)
        self.assertIn(GestureType.FIST, resolved_gestures)
        
    def test_mutually_exclusive_groups(self):
        """Test that mutually exclusive gestures are properly handled."""
        # Create a set with mutually exclusive gestures
        gestures = {GestureType.OPEN_PALM, GestureType.FIST, GestureType.VICTORY}
        
        # Set confidence scores for testing
        self.detector.gesture_confidences[GestureType.OPEN_PALM] = 0.7
        self.detector.gesture_confidences[GestureType.FIST] = 0.6
        self.detector.gesture_confidences[GestureType.VICTORY] = 0.5
        
        # Resolve conflicts
        resolved_gestures = self.detector._resolve_gesture_conflicts(gestures)
        
        # Verify that only one gesture from the mutually exclusive group remains
        self.assertEqual(len(resolved_gestures), 1)
        self.assertIn(GestureType.OPEN_PALM, resolved_gestures)
        
    def test_priority_based_resolution(self):
        """Test that higher priority gestures are preferred during conflict resolution."""
        # Create a set with gestures of different priorities
        gestures = {GestureType.PINCH_INDEX, GestureType.OPEN_PALM}
        
        # Set equal confidence scores for testing
        self.detector.gesture_confidences[GestureType.PINCH_INDEX] = 0.7
        self.detector.gesture_confidences[GestureType.OPEN_PALM] = 0.7
        
        # Resolve conflicts
        resolved_gestures = self.detector._resolve_gesture_conflicts(gestures)
        
        # Verify that the higher priority gesture (PINCH_INDEX) is preferred
        self.assertEqual(len(resolved_gestures), 1)
        self.assertIn(GestureType.PINCH_INDEX, resolved_gestures)
        
    def test_confidence_tiebreaker(self):
        """Test that confidence is used as a tiebreaker when priorities are equal."""
        # Create a set with gestures of equal priority
        gestures = {GestureType.PINCH_INDEX, GestureType.PINCH_MIDDLE}
        
        # Set different confidence scores for testing
        self.detector.gesture_confidences[GestureType.PINCH_INDEX] = 0.9
        self.detector.gesture_confidences[GestureType.PINCH_MIDDLE] = 0.7
        
        # Resolve conflicts
        resolved_gestures = self.detector._resolve_gesture_conflicts(gestures)
        
        # Verify that the higher confidence gesture is preferred
        self.assertEqual(len(resolved_gestures), 1)
        self.assertIn(GestureType.PINCH_INDEX, resolved_gestures)
        
    def _create_mock_landmarks_for_pinch_and_open_palm(self):
        """Create mock landmarks that could be interpreted as both pinch and open palm."""
        landmarks = []
        
        # Create 21 landmarks (MediaPipe hand has 21 landmarks)
        for i in range(21):
            landmark = MagicMock()
            landmark.x = 0.5  # Default position
            landmark.y = 0.5
            landmark.z = 0.0
            landmarks.append(landmark)
        
        # Position landmarks to simulate extended fingers
        # Wrist (0)
        landmarks[0].x = 0.5
        landmarks[0].y = 0.8
        
        # Thumb (1-4)
        landmarks[1].x = 0.4
        landmarks[1].y = 0.7
        landmarks[2].x = 0.3
        landmarks[2].y = 0.65
        landmarks[3].x = 0.25
        landmarks[3].y = 0.6
        landmarks[4].x = 0.2
        landmarks[4].y = 0.55
        
        # Index finger (5-8)
        landmarks[5].x = 0.45
        landmarks[5].y = 0.6
        landmarks[6].x = 0.45
        landmarks[6].y = 0.5
        landmarks[7].x = 0.45
        landmarks[7].y = 0.4
        landmarks[8].x = 0.45
        landmarks[8].y = 0.3
        
        # Middle finger (9-12)
        landmarks[9].x = 0.5
        landmarks[9].y = 0.6
        landmarks[10].x = 0.5
        landmarks[10].y = 0.5
        landmarks[11].x = 0.5
        landmarks[11].y = 0.4
        landmarks[12].x = 0.5
        landmarks[12].y = 0.3
        
        # Ring finger (13-16)
        landmarks[13].x = 0.55
        landmarks[13].y = 0.6
        landmarks[14].x = 0.55
        landmarks[14].y = 0.5
        landmarks[15].x = 0.55
        landmarks[15].y = 0.4
        landmarks[16].x = 0.55
        landmarks[16].y = 0.3
        
        # Pinky finger (17-20)
        landmarks[17].x = 0.6
        landmarks[17].y = 0.6
        landmarks[18].x = 0.6
        landmarks[18].y = 0.5
        landmarks[19].x = 0.6
        landmarks[19].y = 0.4
        landmarks[20].x = 0.6
        landmarks[20].y = 0.3
        
        # Position thumb tip close to index fingertip to simulate pinch
        landmarks[4].x = 0.44  # Thumb tip
        landmarks[4].y = 0.31
        
        return landmarks
        
    def _create_mock_landmarks_with_low_confidence(self):
        """Create mock landmarks with ambiguous gesture (low confidence)."""
        landmarks = []
        
        # Create 21 landmarks (MediaPipe hand has 21 landmarks)
        for i in range(21):
            landmark = MagicMock()
            landmark.x = 0.5  # Default position
            landmark.y = 0.5
            landmark.z = 0.0
            landmarks.append(landmark)
            
        # Position landmarks in an ambiguous way
        # This is a simplified version - in a real test, you would position
        # the landmarks to create a specific ambiguous gesture
        
        return landmarks


if __name__ == '__main__':
    unittest.main()

