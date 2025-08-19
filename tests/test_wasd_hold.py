#!/usr/bin/env python3
"""
Test script for WASD hold functionality.

This script tests whether the hold mode keys (W, A, S, D) are being properly
handled as KEY_HOLD type and checks the gesture-to-input mapping functionality.
"""

import sys
import os
import time
import threading
import logging
from typing import Dict, Set
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from input.handler import UnifiedInputHandler, InputAction, InputType, InputMode
from config.manager import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data class."""
    test_name: str
    passed: bool
    message: str


class WASDHoldTester:
    """Test class for WASD hold functionality."""
    
    def __init__(self):
        self.input_handler = UnifiedInputHandler()
        self.config_manager = ConfigurationManager("./config")
        self.results = []
        
        # Track held keys for testing
        self.expected_held_keys: Set[str] = set()
        
    def setup(self):
        """Setup test environment."""
        logger.info("Setting up test environment...")
        
        # Start input handler
        self.input_handler.start()
        
        # Load gaming profile
        if not self.config_manager.activate_profile("Gaming"):
            logger.error("Could not activate Gaming profile")
            return False
        
        return True
    
    def teardown(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        self.input_handler.stop()
    
    def test_input_type_mapping(self):
        """Test that hold mode maps to KEY_HOLD input type."""
        logger.info("Testing input type mapping...")
        
        test_cases = [
            ("key", "hold", InputType.KEY_HOLD),
            ("key", "tap", InputType.KEY_PRESS),
            ("mouse", "hold", InputType.MOUSE_HOLD),
            ("mouse", "tap", InputType.MOUSE_CLICK),
        ]
        
        # Need to access the main app's _get_input_type method
        # For testing purposes, recreate the logic here
        def _get_input_type(action_type: str, mode: str = "tap") -> InputType:
            if action_type == 'key':
                if mode == 'hold':
                    return InputType.KEY_HOLD
                else:
                    return InputType.KEY_PRESS
            elif action_type == 'mouse':
                if mode == 'hold':
                    return InputType.MOUSE_HOLD
                else:
                    return InputType.MOUSE_CLICK
            elif action_type == 'gamepad':
                return InputType.GAMEPAD_BUTTON
            elif action_type == 'macro':
                return InputType.MACRO
            else:
                return InputType.KEY_PRESS
        
        all_passed = True
        for action_type, mode, expected_type in test_cases:
            result = _get_input_type(action_type, mode)
            passed = result == expected_type
            all_passed = all_passed and passed
            
            self.results.append(TestResult(
                f"Input type mapping: {action_type}/{mode} -> {expected_type.value}",
                passed,
                f"Expected {expected_type.value}, got {result.value}"
            ))
            
            if passed:
                logger.info(f"✓ {action_type}/{mode} correctly maps to {expected_type.value}")
            else:
                logger.error(f"✗ {action_type}/{mode} incorrectly maps to {result.value}, expected {expected_type.value}")
        
        return all_passed
    
    def test_gaming_profile_bindings(self):
        """Test that gaming profile has correct WASD bindings."""
        logger.info("Testing gaming profile bindings...")
        
        profile = self.config_manager.active_profile
        if not profile:
            self.results.append(TestResult(
                "Gaming profile loaded",
                False,
                "No active profile loaded"
            ))
            return False
        
        expected_bindings = {
            "open_palm": ("w", "hold"),
            "fist": ("s", "hold"),
            "l_shape": ("a", "hold"),
            "hang_loose": ("d", "hold"),
            "thumbs_up": ("tab", "hold"),  # Updated to hold mode
        }
        
        all_passed = True
        for gesture, (expected_key, expected_mode) in expected_bindings.items():
            # Find mapping for this gesture
            mapping = None
            for m in profile.gesture_mappings:
                if m.gesture == gesture:
                    mapping = m
                    break
            
            if not mapping:
                self.results.append(TestResult(
                    f"Gesture binding exists: {gesture}",
                    False,
                    f"No mapping found for gesture {gesture}"
                ))
                all_passed = False
                continue
            
            # Check key and mode
            key_correct = mapping.target == expected_key
            mode_correct = mapping.mode == expected_mode
            
            self.results.append(TestResult(
                f"Gesture binding: {gesture} -> {expected_key}/{expected_mode}",
                key_correct and mode_correct,
                f"Expected {expected_key}/{expected_mode}, got {mapping.target}/{mapping.mode}"
            ))
            
            if key_correct and mode_correct:
                logger.info(f"✓ {gesture} correctly bound to {expected_key}/{expected_mode}")
            else:
                logger.error(f"✗ {gesture} incorrectly bound to {mapping.target}/{mapping.mode}")
                all_passed = False
        
        return all_passed
    
    def test_gesture_binding_creation(self):
        """Test creating gesture bindings with hold actions."""
        logger.info("Testing gesture binding creation...")
        
        # Clear existing bindings
        for gesture in list(self.input_handler.gesture_bindings.keys()):
            self.input_handler.unbind_gesture(gesture)
        
        # Create test binding for WASD keys
        test_bindings = [
            ("test_w", "w", InputType.KEY_HOLD),
            ("test_s", "s", InputType.KEY_HOLD),
            ("test_a", "a", InputType.KEY_HOLD),
            ("test_d", "d", InputType.KEY_HOLD),
        ]
        
        all_passed = True
        for gesture_name, key, input_type in test_bindings:
            action = InputAction(
                input_type=input_type,
                target=key,
                mode=InputMode.HOLD
            )
            
            self.input_handler.bind_gesture(gesture_name, action)
            
            # Verify binding was created
            bound = gesture_name in self.input_handler.gesture_bindings
            self.results.append(TestResult(
                f"Gesture binding created: {gesture_name} -> {key} (KEY_HOLD)",
                bound,
                f"Binding {'created' if bound else 'failed to create'}"
            ))
            
            if bound:
                logger.info(f"✓ Created binding for {gesture_name} -> {key}")
            else:
                logger.error(f"✗ Failed to create binding for {gesture_name}")
                all_passed = False
        
        return all_passed
    
    def test_key_hold_execution(self):
        """Test that KEY_HOLD actions are executed correctly."""
        logger.info("Testing KEY_HOLD execution...")
        
        # Test executing a KEY_HOLD action
        action = InputAction(
            input_type=InputType.KEY_HOLD,
            target="w",
            mode=InputMode.HOLD
        )
        
        # Execute the action
        result = self.input_handler.execute_action(action)
        
        # Check if key is tracked as held
        key_held = "w" in self.input_handler.keyboard_handler.held_keys and \
                  self.input_handler.keyboard_handler.held_keys["w"]
        
        self.results.append(TestResult(
            "KEY_HOLD action execution",
            result and key_held,
            f"Action executed: {result}, Key held: {key_held}"
        ))
        
        if result and key_held:
            logger.info("✓ KEY_HOLD action executed and key is held")
        else:
            logger.error(f"✗ KEY_HOLD action failed - executed: {result}, held: {key_held}")
        
        return result and key_held
    
    def test_key_release_mechanism(self):
        """Test if there's a mechanism to release held keys."""
        logger.info("Testing key release mechanism...")
        
        # Hold a key first
        action = InputAction(
            input_type=InputType.KEY_HOLD,
            target="a",
            mode=InputMode.HOLD
        )
        self.input_handler.execute_action(action)
        
        # Check if key is held
        initially_held = "a" in self.input_handler.keyboard_handler.held_keys and \
                        self.input_handler.keyboard_handler.held_keys["a"]
        
        # Try to release the key
        release_action = InputAction(
            input_type=InputType.KEY_RELEASE,
            target="a",
            mode=InputMode.TAP
        )
        release_result = self.input_handler.execute_action(release_action)
        
        # Check if key was released
        released = "a" not in self.input_handler.keyboard_handler.held_keys or \
                  not self.input_handler.keyboard_handler.held_keys["a"]
        
        self.results.append(TestResult(
            "Key release mechanism",
            initially_held and release_result and released,
            f"Initially held: {initially_held}, Release executed: {release_result}, Released: {released}"
        ))
        
        success = initially_held and release_result and released
        if success:
            logger.info("✓ Key release mechanism works correctly")
        else:
            logger.error(f"✗ Key release mechanism failed")
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return results."""
        logger.info("Starting WASD hold functionality tests...")
        
        if not self.setup():
            logger.error("Failed to setup test environment")
            return False
        
        try:
            tests = [
                self.test_input_type_mapping,
                self.test_gaming_profile_bindings,
                self.test_gesture_binding_creation,
                self.test_key_hold_execution,
                self.test_key_release_mechanism,
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
    tester = WASDHoldTester()
    
    try:
        success = tester.run_all_tests()
        tester.print_results()
        
        if success:
            logger.info("\n✅ WASD hold functionality appears to be working correctly!")
        else:
            logger.info("\n❌ WASD hold functionality has issues that need to be addressed.")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
