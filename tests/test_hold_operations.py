#!/usr/bin/env python3
"""
Test script for hold operation fixes.

This script tests the fixes for the issue where gesture count continues to increase
and keys are sent during Hold operations.
"""

import sys
import os
import time
import threading
import logging
from typing import Dict, Set, List
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.input.handler import UnifiedInputHandler, InputAction, InputType, InputMode
from src.gestures.detector import GestureType, GestureDetection
from src.utils.stubs import StatisticsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data class."""
    test_name: str
    passed: bool
    message: str


class HoldOperationTester:
    """Test class for hold operation fixes."""
    
    def __init__(self):
        self.input_handler = UnifiedInputHandler()
        self.stats_tracker = StatisticsTracker()
        self.results = []
        
        # Set stats_tracker in input_handler for duration tracking
        self.input_handler.stats_tracker = self.stats_tracker
        
        # Track test state
        self.active_gestures = set()
        self.gesture_counts = {}
        
    def setup(self):
        """Setup test environment."""
        logger.info("Setting up test environment...")
        
        # Start input handler
        self.input_handler.start()
        
        # Create test bindings for hold operations
        test_bindings = [
            ("open_palm", "w", InputType.KEY_HOLD),
            ("fist", "s", InputType.KEY_HOLD),
            ("l_shape", "a", InputType.KEY_HOLD),
            ("hang_loose", "d", InputType.KEY_HOLD),
        ]
        
        for gesture_name, key, input_type in test_bindings:
            action = InputAction(
                input_type=input_type,
                target=key,
                mode=InputMode.HOLD
            )
            self.input_handler.bind_gesture(gesture_name, action)
            logger.info(f"Bound gesture {gesture_name} to {key} with {input_type.value}")
        
        return True
    
    def teardown(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        self.input_handler.stop()
    
    def test_hold_gesture_count(self):
        """Test that hold gestures don't increment count continuously."""
        logger.info("Testing hold gesture count...")
        
        # Simulate detecting a hold gesture for multiple frames
        gesture_name = "open_palm"
        
        # Initial detection - should execute and increment count
        self._simulate_gesture_detection(gesture_name)
        initial_count = self.gesture_counts.get(gesture_name, 0)
        
        # Simulate multiple frames of the same gesture
        for _ in range(10):
            self._simulate_gesture_detection(gesture_name)
        
        # Check if count increased
        final_count = self.gesture_counts.get(gesture_name, 0)
        count_increased = final_count > initial_count
        
        # For hold gestures, count should not increase after initial detection
        expected_result = not count_increased
        
        self.results.append(TestResult(
            "Hold gesture count stability",
            expected_result,
            f"Initial count: {initial_count}, Final count: {final_count}, Expected no increase: {expected_result}"
        ))
        
        if expected_result:
            logger.info(f"✓ Hold gesture count remained stable at {final_count}")
        else:
            logger.error(f"✗ Hold gesture count increased from {initial_count} to {final_count}")
        
        return expected_result
    
    def test_hold_key_execution(self):
        """Test that hold gestures don't send keys repeatedly."""
        logger.info("Testing hold key execution...")
        
        # Simulate detecting a hold gesture for multiple frames
        gesture_name = "fist"
        
        # Track key executions
        key_executions = []
        
        # Monkey patch execute_action to track executions
        original_execute_action = self.input_handler.execute_action
        
        def tracked_execute_action(action):
            key_executions.append(action.target)
            return original_execute_action(action)
        
        self.input_handler.execute_action = tracked_execute_action
        
        try:
            # Initial detection - should execute
            self._simulate_gesture_detection(gesture_name)
            initial_executions = len(key_executions)
            
            # Simulate multiple frames of the same gesture
            for _ in range(10):
                self._simulate_gesture_detection(gesture_name)
            
            # Check if additional executions occurred
            final_executions = len(key_executions)
            
            # For hold gestures, only one key press should occur
            # and one key release when deactivated
            expected_result = final_executions == initial_executions
            
            self.results.append(TestResult(
                "Hold key execution stability",
                expected_result,
                f"Initial executions: {initial_executions}, Final executions: {final_executions}, Expected no increase: {expected_result}"
            ))
            
            if expected_result:
                logger.info(f"✓ Hold key executions remained stable at {final_executions}")
            else:
                logger.error(f"✗ Hold key executions increased from {initial_executions} to {final_executions}")
            
            return expected_result
            
        finally:
            # Restore original method
            self.input_handler.execute_action = original_execute_action
    
    def test_hold_release_mechanism(self):
        """Test that hold gestures properly release keys when deactivated."""
        logger.info("Testing hold release mechanism...")
        
        # Simulate detecting a hold gesture
        gesture_name = "l_shape"
        
        # Track key presses and releases
        key_actions = []
        
        # Monkey patch execute_action to track actions
        original_execute_action = self.input_handler.execute_action
        
        def tracked_execute_action(action):
            key_actions.append((action.input_type.value, action.target))
            return original_execute_action(action)
        
        self.input_handler.execute_action = tracked_execute_action
        
        try:
            # Activate the gesture
            self._simulate_gesture_detection(gesture_name)
            
            # Deactivate the gesture
            self._simulate_gesture_deactivation(gesture_name)
            
            # Check if both press and release occurred
            has_press = any(action[0] == InputType.KEY_HOLD.value and action[1] == "a" for action in key_actions)
            has_release = any(action[0] == InputType.KEY_RELEASE.value and action[1] == "a" for action in key_actions)
            
            expected_result = has_press and has_release
            
            self.results.append(TestResult(
                "Hold release mechanism",
                expected_result,
                f"Has press: {has_press}, Has release: {has_release}, Expected both: {expected_result}"
            ))
            
            if expected_result:
                logger.info(f"✓ Hold gesture properly pressed and released key")
            else:
                logger.error(f"✗ Hold gesture failed to properly press and/or release key")
            
            return expected_result
            
        finally:
            # Restore original method
            self.input_handler.execute_action = original_execute_action
    
    def test_active_gesture_tracking(self):
        """Test that active gestures are properly tracked."""
        logger.info("Testing active gesture tracking...")
        
        # Clear active gestures
        self.active_gestures.clear()
        self.input_handler.active_gestures.clear()
        
        # Simulate detecting multiple gestures
        gestures = ["open_palm", "fist", "l_shape"]
        
        for gesture_name in gestures:
            self._simulate_gesture_detection(gesture_name)
        
        # Check if all gestures are tracked as active
        all_active = all(gesture_name in self.input_handler.active_gestures for gesture_name in gestures)
        
        # Deactivate one gesture
        self._simulate_gesture_deactivation("fist")
        
        # Check if deactivated gesture is removed
        fist_removed = "fist" not in self.input_handler.active_gestures
        
        expected_result = all_active and fist_removed
        
        self.results.append(TestResult(
            "Active gesture tracking",
            expected_result,
            f"All gestures active: {all_active}, Deactivated gesture removed: {fist_removed}, Expected both: {expected_result}"
        ))
        
        if expected_result:
            logger.info(f"✓ Active gestures properly tracked and updated")
        else:
            logger.error(f"✗ Active gesture tracking failed")
        
        return expected_result
    
    def _simulate_gesture_detection(self, gesture_name: str):
        """Simulate detecting a gesture."""
        # Update active gestures
        self.active_gestures.add(gesture_name)
        self.input_handler.update_active_gestures(self.active_gestures)
        
        # Create a detection
        detection = GestureDetection(
            gesture_type=GestureType(gesture_name),
            confidence=0.9,
            timestamp=time.time()
        )
        
        # Handle the detection
        self._handle_gesture(detection)
    
    def _simulate_gesture_deactivation(self, gesture_name: str):
        """Simulate a gesture being deactivated."""
        # Remove from active gestures
        if gesture_name in self.active_gestures:
            self.active_gestures.remove(gesture_name)
        
        # Update active gestures
        self.input_handler.update_active_gestures(self.active_gestures)
    
    def _handle_gesture(self, detection):
        """Simulate the _handle_gesture method from main.py."""
        gesture_name = detection.gesture_type.value
        
        # Check if gesture is already active
        if gesture_name in self.input_handler.active_gestures:
            # Check if this gesture is in hold mode
            is_hold_gesture = gesture_name in self.input_handler.gesture_hold_states
            
            # For hold gestures, we don't even update statistics
            if is_hold_gesture:
                logger.debug(f"Skipping already active hold gesture: {gesture_name}")
                return
            
            # For non-hold gestures, update continuation statistics
            confidence = getattr(detection, 'confidence', 1.0)
            self.stats_tracker.record_gesture_continuation(gesture_name, confidence)
            return
        
        # Gesture is new - execute action
        logger.debug(f"Executing new gesture: {gesture_name}")
        success = self.input_handler.execute_gesture(gesture_name)
        
        # Start tracking gesture duration
        if success:
            self.stats_tracker.start_gesture_duration(gesture_name)
        
        # Update statistics
        confidence = getattr(detection, 'confidence', 1.0)
        self.stats_tracker.record_gesture(gesture_name, success, confidence)
        
        # Update gesture counts for testing
        if gesture_name not in self.gesture_counts:
            self.gesture_counts[gesture_name] = 0
        self.gesture_counts[gesture_name] += 1
    
    def run_all_tests(self):
        """Run all tests and return results."""
        logger.info("Starting hold operation tests...")
        
        if not self.setup():
            logger.error("Failed to setup test environment")
            return False
        
        try:
            tests = [
                self.test_hold_gesture_count,
                self.test_hold_key_execution,
                self.test_hold_release_mechanism,
                self.test_active_gesture_tracking,
            ]
            
            all_passed = True
            for test in tests:
                try:
                    result = test()
                    all_passed = all_passed and result
                except Exception as e:
                    logger.error(f"Test {test.__name__} failed with exception: {e}")
                    self.results.append(TestResult(
                        test.__name__,
                        False,
                        f"Exception: {str(e)}"
                    ))
                    all_passed = False
                
                # Small delay between tests
                time.sleep(0.1)
            
            return all_passed
        
        finally:
            self.teardown()
    
    def print_results(self):
        """Print test results."""
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)
        
        passed_count = 0
        total_count = len(self.results)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            logger.info(f"[{status}] {result.test_name}")
            if not result.passed:
                logger.info(f"        {result.message}")
            else:
                passed_count += 1
        
        logger.info("="*60)
        logger.info(f"SUMMARY: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.info("❌ Some tests failed. Please check the implementation.")
        
        return passed_count == total_count


def main():
    """Main test function."""
    tester = HoldOperationTester()
    
    try:
        success = tester.run_all_tests()
        tester.print_results()
        
        if success:
            logger.info("\n✅ Hold operation fixes appear to be working correctly!")
        else:
            logger.info("\n❌ Hold operation fixes have issues that need to be addressed.")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

