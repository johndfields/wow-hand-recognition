#!/usr/bin/env python3
"""
Test script to verify that gesture filtering optimization is working correctly.
This tests that disabled gestures are not processed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.manager import ConfigurationManager
from gestures.detector import StandardGestureDetector, GestureProcessor, GestureType

def test_gesture_filtering():
    """Test that gesture filtering works correctly"""
    print("=== Testing Gesture Filtering Optimization ===\n")
    
    # Load the Gaming profile
    config_manager = ConfigurationManager('config')
    config_manager.activate_profile('Gaming')
    
    print(f"Active profile: {config_manager.active_profile.name}")
    print(f"Profile has {len(config_manager.active_profile.gesture_mappings)} gesture mappings\n")
    
    # List the enabled gestures from the config
    enabled_gestures_from_config = set()
    for mapping in config_manager.active_profile.gesture_mappings:
        if mapping.enabled:
            enabled_gestures_from_config.add(mapping.gesture)
    
    print("Enabled gestures from Gaming profile config:")
    for gesture in sorted(enabled_gestures_from_config):
        print(f"  ✓ {gesture}")
    
    print(f"\nDisabled gestures (not in profile):")
    all_gestures = {gesture.value for gesture in GestureType}
    disabled_gestures = all_gestures - enabled_gestures_from_config
    for gesture in sorted(disabled_gestures):
        print(f"  ✗ {gesture}")
    
    print(f"\n🔍 Specifically checking 'thumbs_up': {'✓ ENABLED' if 'thumbs_up' in enabled_gestures_from_config else '✗ DISABLED'}")
    
    # Create a detector with the enabled gestures
    detector = StandardGestureDetector(
        sensitivity=1.0,
        enable_motion_gestures=True,
        enable_custom_gestures=False,
        enabled_gestures=enabled_gestures_from_config
    )
    
    print(f"\nDetector enabled gestures ({len(detector.enabled_gestures)}):")
    for gesture_type in sorted(detector.enabled_gestures, key=lambda x: x.value):
        print(f"  ✓ {gesture_type.value}")
    
    print(f"\nDetector disabled gestures:")
    all_gesture_types = set(GestureType)
    disabled_gesture_types = all_gesture_types - detector.enabled_gestures
    for gesture_type in sorted(disabled_gesture_types, key=lambda x: x.value):
        print(f"  ✗ {gesture_type.value}")
    
    # Test GestureProcessor with config manager
    print("\n" + "="*50)
    print("Testing GestureProcessor with config integration:")
    
    processor = GestureProcessor(
        detector=detector,
        enable_multi_hand=False,
        max_hands=1,
        confidence_threshold=0.6,
        hand_preference="right",
        config_manager=config_manager
    )
    
    print(f"GestureProcessor detector enabled gestures ({len(processor.detector.enabled_gestures)}):")
    for gesture_type in sorted(processor.detector.enabled_gestures, key=lambda x: x.value):
        print(f"  ✓ {gesture_type.value}")
    
    # Verify thumbs_up specifically
    thumbs_up_enabled = GestureType.THUMBS_UP in processor.detector.enabled_gestures
    print(f"\n🎯 CRITICAL CHECK - thumbs_up detection:")
    print(f"   In Gaming profile config: {'YES' if 'thumbs_up' in enabled_gestures_from_config else 'NO'}")
    print(f"   In detector enabled_gestures: {'YES' if thumbs_up_enabled else 'NO'}")
    
    if thumbs_up_enabled:
        print("   ⚠️  WARNING: thumbs_up should be DISABLED in Gaming profile!")
    else:
        print("   ✅ CORRECT: thumbs_up is properly disabled")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"- Gaming profile has {len(enabled_gestures_from_config)} enabled gestures")
    print(f"- Detector is configured with {len(detector.enabled_gestures)} enabled gesture types")
    print(f"- thumbs_up is {'ENABLED' if thumbs_up_enabled else 'DISABLED'} in detector")
    
    if thumbs_up_enabled:
        print("\n❌ ISSUE FOUND: thumbs_up should be disabled but is enabled")
        return False
    else:
        print("\n✅ OPTIMIZATION WORKING: thumbs_up is properly disabled")
        return True

if __name__ == "__main__":
    try:
        success = test_gesture_filtering()
        if success:
            print("\n🎉 Gesture filtering optimization is working correctly!")
        else:
            print("\n🚨 Gesture filtering optimization needs debugging!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
