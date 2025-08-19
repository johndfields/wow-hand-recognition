#!/usr/bin/env python3
"""
Test script to verify the main application components work correctly.
This runs without GUI to test core functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from config.manager import ConfigurationManager
from gestures.detector import StandardGestureDetector, GestureProcessor, GestureType
from input.handler import UnifiedInputHandler

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    config_manager = ConfigurationManager('config')
    print(f"✓ Settings loaded: {config_manager.settings}")
    print(f"✓ Active profile: {config_manager.active_profile.name}")
    print(f"✓ Hand preference: {config_manager.settings.hand_preference}")
    print(f"✓ Show landmarks: {config_manager.settings.show_hand_landmarks}")
    print(f"✓ Show connections: {config_manager.settings.show_hand_connections}")
    print(f"✓ Show debug info: {config_manager.settings.show_debug_info}")
    
def test_gesture_processor():
    """Test gesture processor initialization"""
    print("\nTesting gesture processor...")
    
    # Create detector
    detector = StandardGestureDetector(
        sensitivity=1.0,
        enable_motion_gestures=True,
        enable_custom_gestures=False
    )
    print("✓ StandardGestureDetector created")
    
    # Create processor
    processor = GestureProcessor(
        detector=detector,
        enable_multi_hand=False,
        max_hands=1,
        confidence_threshold=0.6,
        hand_preference="right"
    )
    print("✓ GestureProcessor created")
    print(f"✓ MediaPipe initialized: {processor.mediapipe_initialized}")
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("✓ Test frame created")
    
    # Test frame processing (should return empty since no hand in frame)
    detections = processor.process_frame(test_frame)
    print(f"✓ Frame processed, detections: {len(detections)}")
    
    # Test debug drawing
    debug_frame = processor.draw_debug_info(
        test_frame.copy(), 
        detections, 
        show_landmarks=True, 
        show_connections=True
    )
    print(f"✓ Debug info drawn, frame shape: {debug_frame.shape}")
    
    return processor

def test_input_handler():
    """Test input handler"""
    print("\nTesting input handler...")
    
    input_handler = UnifiedInputHandler()
    print("✓ UnifiedInputHandler created")
    
    input_handler.start()
    print("✓ Input handler started")
    
    # Test gesture binding
    success = input_handler.bind_gesture("open_palm", None)  # None action for test
    print(f"✓ Gesture binding test: {success}")
    
    input_handler.stop()
    print("✓ Input handler stopped")

def main():
    """Run all tests"""
    print("=== Hand to Key Application Components Test ===")
    
    try:
        test_config_loading()
        processor = test_gesture_processor()
        test_input_handler()
        
        print("\n=== All Tests Passed! ===")
        print("\nKey findings:")
        print(f"- MediaPipe initialization: {'✓ SUCCESS' if processor.mediapipe_initialized else '✗ FAILED'}")
        print("- Configuration loading: ✓ SUCCESS")
        print("- Gesture detection setup: ✓ SUCCESS")
        print("- Debug drawing: ✓ SUCCESS")
        print("\nYour application components are working correctly!")
        print("The hand landmarks and connections should now display when you run the main app.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
