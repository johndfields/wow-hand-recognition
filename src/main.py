#!/usr/bin/env python3
"""
Hand to Key - Advanced Hand Gesture Control System
Main application with all enhancements integrated.
"""

import sys
import os
import argparse
import logging
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from dataclasses import dataclass
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gestures.detector import (
    GestureProcessor, StandardGestureDetector, CustomGestureDetector,
    GestureType, GestureDetection
)
from input.handler import (
    UnifiedInputHandler, InputAction, InputSequence,
    InputType, InputMode
)
from config.manager import (
    ConfigurationManager, GestureMapping
)
from utils.camera import CameraManager, CameraSelector
# Import utility modules
from utils.stubs import (
    AudioFeedback, StatisticsTracker, CrashHandler,
    OverlayRenderer, CalibrationMode, VoiceCommandHandler,
    SystemTrayApp
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ApplicationState:
    """Maintains application state."""
    is_running: bool = True
    is_paused: bool = False
    is_recording: bool = False
    is_calibrating: bool = False
    current_profile: str = ""
    gesture_history: List[GestureDetection] = None
    performance_mode: str = "balanced"  # low, balanced, high
    
    def __post_init__(self):
        if self.gesture_history is None:
            self.gesture_history = []


class HandToKeyApplication:
    """Main application class with all enhancements."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = ApplicationState()
        
        # Initialize components
        self._initialize_config()
        self._initialize_camera()
        self._initialize_gesture_processor()
        self._initialize_input_handler()
        self._initialize_ui()
        self._initialize_audio()
        self._initialize_accessibility()
        self._initialize_platform_integration()
        self._initialize_crash_handler()
        
        # Statistics
        self.stats_tracker = StatisticsTracker()
        
        # Threading
        self.main_thread = threading.current_thread()
        self.processing_queue = queue.Queue(maxsize=60)
        self.shutdown_event = threading.Event()
        
        logger.info("Hand to Key application initialized")
    
    def _initialize_config(self):
        """Initialize configuration management."""
        config_dir = self.args.config_dir or "./config"
        self.config_manager = ConfigurationManager(config_dir)
        
        # Create default profiles if none exist
        if not self.config_manager.profiles:
            self.config_manager.create_default_profiles()
        
        # Load or activate profile
        if self.args.profile:
            self.config_manager.activate_profile(self.args.profile)
        elif not self.config_manager.active_profile:
            profiles = self.config_manager.get_all_profiles()
            # Prefer Gaming profile as default, fall back to first available
            if "Gaming" in profiles:
                self.config_manager.activate_profile("Gaming")
            elif profiles:
                self.config_manager.activate_profile(profiles[0])
        
        # Start hot-reload if enabled
        if self.args.hot_reload:
            self.config_manager.start_hot_reload()
            self.config_manager.add_reload_callback(self._on_config_reload)
        
        # Update state
        if self.config_manager.active_profile:
            self.state.current_profile = self.config_manager.active_profile.name
    
    def _initialize_camera(self):
        """Initialize camera with auto-selection and adaptive quality."""
        settings = self.config_manager.settings
        
        # Auto-select camera if needed
        if self.args.auto_camera:
            selector = CameraSelector()
            camera_index = selector.select_best_camera()
            logger.info(f"Auto-selected camera index: {camera_index}")
        else:
            camera_index = self.args.camera or settings.camera_index
        
        # Initialize camera manager
        self.camera_manager = CameraManager(
            camera_index=camera_index,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            adaptive_quality=settings.adaptive_quality
        )
        
        if not self.camera_manager.start():
            logger.error("Failed to initialize camera")
            sys.exit(1)
    
    def _initialize_gesture_processor(self):
        """Initialize gesture detection and processing."""
        settings = self.config_manager.settings
        
        # Create detector
        if self.args.custom_gestures:
            detector = CustomGestureDetector(self.args.model_path)
        else:
            detector = StandardGestureDetector(
                sensitivity=settings.gesture_sensitivity,
                enable_motion_gestures=True,
                enable_custom_gestures=False
            )
        
        # Create processor with optimizations
        self.gesture_processor = GestureProcessor(
            detector=detector,
            enable_multi_hand=settings.enable_multi_hand,
            max_hands=settings.max_hands,
            enable_gpu=settings.enable_gpu,
            frame_skip=settings.frame_skip,
            confidence_threshold=settings.min_detection_confidence,
            temporal_smoothing=settings.enable_temporal_smoothing,
            smoothing_window=settings.smoothing_window,
            min_gesture_frames=settings.min_gesture_frames,
            hand_preference=settings.hand_preference,
            config_manager=self.config_manager
        )
        
        # Start processing thread if threading enabled
        if self.args.threading:
            self.gesture_processor.start()
    
    def _initialize_input_handler(self):
        """Initialize unified input handler."""
        self.input_handler = UnifiedInputHandler()
        self.input_handler.start()
        
        # Bind gestures from active profile
        self._update_gesture_bindings()
    
    def _initialize_ui(self):
        """Initialize UI components."""
        settings = self.config_manager.settings
        
        # Create overlay renderer
        self.overlay_renderer = OverlayRenderer(
            show_statistics=settings.show_statistics,
            show_debug_info=settings.show_debug_info,
            ui_scale=settings.ui_scale,
            theme=settings.theme,
            config_manager=self.config_manager
        )
        
        # Create calibration mode
        self.calibration_mode = CalibrationMode(
            gesture_processor=self.gesture_processor,
            overlay_renderer=self.overlay_renderer
        )
    
    def _initialize_audio(self):
        """Initialize audio feedback system."""
        settings = self.config_manager.settings
        
        if settings.enable_sound_feedback:
            self.audio_feedback = AudioFeedback(
                volume=settings.sound_volume,
                enabled=True
            )
        else:
            self.audio_feedback = None
    
    def _initialize_accessibility(self):
        """Initialize accessibility features."""
        settings = self.config_manager.settings
        
        if settings.voice_commands:
            self.voice_handler = VoiceCommandHandler(
                command_callback=self._handle_voice_command
            )
            self.voice_handler.start()
        else:
            self.voice_handler = None
        
        # Apply accessibility settings
        if settings.accessibility_mode:
            self._apply_accessibility_settings()
    
    def _initialize_platform_integration(self):
        """Initialize platform-specific integrations."""
        if self.args.system_tray:
            self.system_tray = SystemTrayApp(
                app_controller=self,
                profiles=self.config_manager.get_all_profiles()
            )
            threading.Thread(target=self.system_tray.run, daemon=True).start()
        else:
            self.system_tray = None
    
    def _initialize_crash_handler(self):
        """Initialize crash recovery system."""
        self.crash_handler = CrashHandler(
            app_restart_callback=self.restart,
            state_save_callback=self.save_state,
            state_load_callback=self.load_state
        )
        self.crash_handler.register()
    
    def _update_gesture_bindings(self):
        """Update gesture bindings from configuration."""
        if not self.config_manager.active_profile:
            return
        
        # Clear existing bindings
        for gesture in list(self.input_handler.gesture_bindings.keys()):
            self.input_handler.unbind_gesture(gesture)
        
        # Add new bindings
        for mapping in self.config_manager.active_profile.gesture_mappings:
            if not mapping.enabled:
                continue
            
            # Create input action
            input_type = self._get_input_type(mapping.action_type, mapping.mode)
            input_mode = self._get_input_mode(mapping.mode)
            
            action = InputAction(
                input_type=input_type,
                target=mapping.target,
                mode=input_mode,
                duration=mapping.duration,
                metadata=mapping.metadata
            )
            
            self.input_handler.bind_gesture(mapping.gesture, action)
            logger.debug(f"Bound gesture '{mapping.gesture}' to {mapping.action_type}:{mapping.target}")
    
    def _get_input_type(self, action_type: str, mode: str = "tap") -> InputType:
        """Convert action type and mode to InputType enum."""
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
    
    def _get_input_mode(self, mode: str) -> InputMode:
        """Convert mode string to InputMode enum."""
        mode_map = {
            'tap': InputMode.TAP,
            'hold': InputMode.HOLD,
            'toggle': InputMode.TOGGLE,
            'double_tap': InputMode.DOUBLE_TAP,
            'long_press': InputMode.LONG_PRESS
        }
        return mode_map.get(mode, InputMode.TAP)
    
    def _on_config_reload(self, file_path: Any):
        """Handle configuration reload event."""
        logger.info(f"Configuration reloaded: {file_path}")
        
        # Update enabled gestures in the gesture processor
        if hasattr(self, 'gesture_processor'):
            self.gesture_processor.update_enabled_gestures_from_config()
        
        # Only update gesture bindings if input handler is initialized
        if hasattr(self, 'input_handler'):
            self._update_gesture_bindings()
        
        # Update settings
        settings = self.config_manager.settings
        if hasattr(self, 'audio_feedback') and self.audio_feedback:
            self.audio_feedback.set_volume(settings.sound_volume)
            self.audio_feedback.set_enabled(settings.enable_sound_feedback)
    
    def _handle_voice_command(self, command: str):
        """Handle voice command."""
        logger.info(f"Voice command: {command}")
        
        # Parse and execute command
        if "pause" in command.lower():
            self.toggle_pause()
        elif "resume" in command.lower():
            self.resume()
        elif "profile" in command.lower():
            # Extract profile name and switch
            words = command.lower().split()
            if "profile" in words:
                idx = words.index("profile")
                if idx + 1 < len(words):
                    profile_name = words[idx + 1]
                    self.switch_profile(profile_name)
        elif "calibrate" in command.lower():
            self.start_calibration()
        elif "stop" in command.lower() or "quit" in command.lower():
            self.stop()
    
    def _apply_accessibility_settings(self):
        """Apply accessibility settings."""
        settings = self.config_manager.settings
        
        if settings.high_contrast:
            self.overlay_renderer.enable_high_contrast()
        
        if settings.large_text:
            self.overlay_renderer.set_text_scale(1.5)
        
        # Adjust gesture sensitivity for accessibility
        if self.gesture_processor.detector:
            self.gesture_processor.detector.sensitivity = settings.gesture_sensitivity * 1.2
    
    def run(self):
        """Main application loop."""
        logger.info("Starting Hand to Key application")
        
        try:
            while self.state.is_running and not self.shutdown_event.is_set():
                # Check if paused
                if self.state.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                if frame is None:
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)
                
                # Update statistics
                self.stats_tracker.update()
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.crash_handler.handle_crash(e)
        finally:
            self.cleanup()
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame."""
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect gestures
        if self.args.threading:
            self.gesture_processor.process_frame_async(frame)
            detections = self.gesture_processor.get_results() or []
        else:
            detections = self.gesture_processor.process_frame(frame)
        
        # Extract currently detected gesture names for state tracking
        current_gesture_names = {detection.gesture_type.value for detection in detections}
        
        # Process detections
        for detection in detections:
            self._handle_gesture(detection)
        
        # Update gesture state management for hold/release functionality
        self.input_handler.update_active_gestures(current_gesture_names)
        
        # Update gesture history
        self.state.gesture_history.extend(detections)
        if len(self.state.gesture_history) > 100:
            self.state.gesture_history = self.state.gesture_history[-100:]
        
        # Render overlay
        if self.config_manager.settings.show_preview:
            # Draw hand tracking visualization if landmarks or connections are enabled
            if (self.config_manager.settings.show_hand_landmarks or 
                self.config_manager.settings.show_hand_connections or
                self.config_manager.settings.show_debug_info):
                frame = self.gesture_processor.draw_debug_info(
                    frame, 
                    detections,
                    show_landmarks=self.config_manager.settings.show_hand_landmarks,
                    show_connections=self.config_manager.settings.show_hand_connections
                )
            
            # Draw overlay
            frame = self.overlay_renderer.render(
                frame,
                detections=detections,
                stats=self._get_stats(),
                state=self.state
            )
            
            # Show calibration if active
            if self.state.is_calibrating:
                frame = self.calibration_mode.render(frame)
            
            # Display frame
            cv2.imshow("Hand to Key - Enhanced", frame)
    
    def _handle_gesture(self, detection: GestureDetection):
        """Handle a detected gesture with state-change optimization."""
        gesture_name = detection.gesture_type.value
        
        # Check if in calibration mode
        if self.state.is_calibrating:
            self.calibration_mode.record_gesture(detection)
            return
        
        # Check if recording
        if self.state.is_recording:
            # Recording handled separately
            return
        
        # Check if gesture is already active (state hasn't changed)
        if gesture_name in self.input_handler.active_gestures:
            # Gesture is already active - update continuation statistics but don't re-execute
            confidence = getattr(detection, 'confidence', 1.0)
            self.stats_tracker.record_gesture_continuation(gesture_name, confidence)
            return
        
        # Gesture is new or reactivated - execute action
        success = self.input_handler.execute_gesture(gesture_name)
        
        # Play audio feedback only on new activations
        if success and self.audio_feedback:
            self.audio_feedback.play_gesture_sound(gesture_name)
        
        # Update statistics with confidence for new activation
        confidence = getattr(detection, 'confidence', 1.0)
        self.stats_tracker.record_gesture(gesture_name, success, confidence)
        
        # Add notification for successful gestures in detailed mode (only new activations)
        if success and self.overlay_renderer.display_mode in ["detailed", "debug"]:
            action_text = self._get_gesture_action_text(gesture_name)
            if action_text:
                self.overlay_renderer.add_notification(f"{self.overlay_renderer._format_gesture_name(gesture_name)} → {action_text}", 1.0)
    
    def _get_gesture_action_text(self, gesture_name: str) -> str:
        """Get action text for a gesture."""
        if not self.config_manager.active_profile:
            return ""
        
        for mapping in self.config_manager.active_profile.gesture_mappings:
            if mapping.gesture == gesture_name:
                return mapping.target.upper()
        
        return ""
    
    def _handle_keyboard(self, key: int):
        """Handle keyboard input with enhanced overlay controls."""
        if key == ord('q'):
            self.stop()
        elif key == ord('p'):
            self.toggle_pause()
        elif key == ord('c'):
            self.start_calibration()
        elif key == ord('r'):
            self.toggle_recording()
        elif key == ord('s'):
            self.save_settings()
        elif key == ord('h'):
            self.show_help()
        elif key == ord('d'):
            # Toggle display mode
            self.overlay_renderer.toggle_display_mode()
            self.overlay_renderer.add_notification(f"Display mode: {self.overlay_renderer.display_mode.title()}", 2.0)
        elif key == ord('t'):
            # Toggle theme
            current_theme = self.overlay_renderer.theme
            new_theme = "light" if current_theme == "dark" else "dark"
            self.overlay_renderer.set_theme(new_theme)
            self.overlay_renderer.add_notification(f"Theme: {new_theme.title()}", 2.0)
        elif key == ord('m'):
            # Toggle statistics display
            self.overlay_renderer.show_statistics = not self.overlay_renderer.show_statistics
            status = "ON" if self.overlay_renderer.show_statistics else "OFF"
            self.overlay_renderer.add_notification(f"Statistics: {status}", 2.0)
        elif key == ord('k'):
            # Toggle keybindings panel
            self.overlay_renderer.show_keybindings = not self.overlay_renderer.show_keybindings
            status = "ON" if self.overlay_renderer.show_keybindings else "OFF"
            self.overlay_renderer.add_notification(f"Keybindings: {status}", 2.0)
        elif key == ord('=') or key == ord('+'):
            # Increase UI scale
            self.overlay_renderer.ui_scale = min(2.0, self.overlay_renderer.ui_scale + 0.1)
            self.overlay_renderer.add_notification(f"UI Scale: {self.overlay_renderer.ui_scale:.1f}", 1.5)
        elif key == ord('-') or key == ord('_'):
            # Decrease UI scale
            self.overlay_renderer.ui_scale = max(0.5, self.overlay_renderer.ui_scale - 0.1)
            self.overlay_renderer.add_notification(f"UI Scale: {self.overlay_renderer.ui_scale:.1f}", 1.5)
        elif key == ord('l'):
            # Toggle hand landmarks
            self.config_manager.settings.show_hand_landmarks = not self.config_manager.settings.show_hand_landmarks
            status = "ON" if self.config_manager.settings.show_hand_landmarks else "OFF"
            self.overlay_renderer.add_notification(f"Hand Landmarks: {status}", 2.0)
        elif key == ord('j'):
            # Toggle hand connections
            self.config_manager.settings.show_hand_connections = not self.config_manager.settings.show_hand_connections
            status = "ON" if self.config_manager.settings.show_hand_connections else "OFF"
            self.overlay_renderer.add_notification(f"Hand Connections: {status}", 2.0)
        elif key == ord('b'):
            # Cycle through hand preferences: right -> left -> both -> right
            current = self.config_manager.settings.hand_preference
            if current == "right":
                new_pref = "left"
            elif current == "left":
                new_pref = "both"
            else:  # both
                new_pref = "right"
            
            self.config_manager.settings.hand_preference = new_pref
            self.gesture_processor.hand_preference = new_pref
            self.overlay_renderer.add_notification(f"Hand Preference: {new_pref.title()}", 2.0)
            logger.info(f"Hand preference changed to: {new_pref}")
        elif key >= ord('1') and key <= ord('9'):
            # Switch profile by number
            profile_idx = key - ord('1')
            profiles = self.config_manager.get_all_profiles()
            if profile_idx < len(profiles):
                self.switch_profile(profiles[profile_idx])
                self.overlay_renderer.add_notification(f"Switched to: {profiles[profile_idx]}", 2.0)
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {
            'fps': self.camera_manager.get_fps(),
            'gestures_detected': self.stats_tracker.total_gestures,
            'success_rate': self.stats_tracker.get_success_rate(),
            'profile': self.state.current_profile,
            'gesture_stats': self.gesture_processor.get_statistics(),
            'input_stats': self.input_handler.get_statistics()
        }
        return stats
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.state.is_paused = not self.state.is_paused
        logger.info(f"Application {'paused' if self.state.is_paused else 'resumed'}")
        
        if self.audio_feedback:
            self.audio_feedback.play_system_sound('pause' if self.state.is_paused else 'resume')
    
    def resume(self):
        """Resume from pause."""
        self.state.is_paused = False
        logger.info("Application resumed")
    
    def start_calibration(self):
        """Start calibration mode."""
        if not self.state.is_calibrating:
            self.state.is_calibrating = True
            self.calibration_mode.start()
            logger.info("Calibration mode started")
        else:
            self.stop_calibration()
    
    def stop_calibration(self):
        """Stop calibration mode."""
        if self.state.is_calibrating:
            self.state.is_calibrating = False
            results = self.calibration_mode.stop()
            logger.info(f"Calibration completed: {results}")
            
            # Apply calibration results
            if results:
                self._apply_calibration(results)
    
    def _apply_calibration(self, results: Dict[str, Any]):
        """Apply calibration results."""
        # Update gesture sensitivity based on calibration
        if 'sensitivity' in results:
            self.config_manager.settings.gesture_sensitivity = results['sensitivity']
            self.gesture_processor.detector.sensitivity = results['sensitivity']
        
        # Save settings
        self.config_manager.save_settings()
    
    def toggle_recording(self):
        """Toggle macro recording."""
        if not self.state.is_recording:
            self.state.is_recording = True
            self.input_handler.start_recording('all')
            logger.info("Macro recording started")
        else:
            self.state.is_recording = False
            macro_name = f"macro_{int(time.time())}"
            sequence = self.input_handler.stop_recording(macro_name)
            logger.info(f"Macro recording stopped, saved as '{macro_name}' with {len(sequence.actions)} actions")
    
    def switch_profile(self, profile_name: str):
        """Switch to a different profile."""
        if self.config_manager.activate_profile(profile_name):
            self.state.current_profile = profile_name
            # Update enabled gestures in the gesture processor
            if hasattr(self, 'gesture_processor'):
                self.gesture_processor.update_enabled_gestures_from_config()
            self._update_gesture_bindings()
            logger.info(f"Switched to profile: {profile_name}")
            
            if self.audio_feedback:
                self.audio_feedback.play_system_sound('profile_switch')
    
    def save_settings(self):
        """Save current settings."""
        self.config_manager.save_settings()
        if self.config_manager.active_profile:
            self.config_manager.save_profile(self.config_manager.active_profile)
        logger.info("Settings saved")
    
    def save_state(self) -> Dict[str, Any]:
        """Save application state for crash recovery."""
        state = {
            'profile': self.state.current_profile,
            'is_paused': self.state.is_paused,
            'gesture_history': [
                {
                    'gesture': g.gesture_type.value,
                    'confidence': g.confidence,
                    'timestamp': g.timestamp
                }
                for g in self.state.gesture_history[-10:]
            ],
            'stats': self.stats_tracker.get_summary()
        }
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load application state after crash recovery."""
        if 'profile' in state:
            self.switch_profile(state['profile'])
        
        if 'is_paused' in state:
            self.state.is_paused = state['is_paused']
        
        logger.info("State restored after crash recovery")
    
    def restart(self):
        """Restart the application."""
        logger.info("Restarting application...")
        self.cleanup()
        
        # Re-initialize
        self.__init__(self.args)
        self.run()
    
    def show_help(self):
        """Display help information."""
        help_text = """
        Hand to Key - Keyboard Shortcuts:
        
        Basic Controls:
        q - Quit application
        p - Pause/Resume
        h - Show this help
        s - Save settings
        
        Profiles:
        1-9 - Switch to profile by number
        
        Hand Tracking:
        b - Cycle hand preference (right → left → both)
        l - Toggle hand landmarks visualization
        j - Toggle hand connections visualization
        d - Toggle debug/visualization mode
        
        Recording & Calibration:
        c - Start/Stop calibration
        r - Start/Stop macro recording
        
        Display:
        t - Toggle theme (dark/light)
        m - Toggle statistics display
        k - Toggle keybindings panel
        +/= - Increase UI scale
        -/_ - Decrease UI scale
        """
        logger.info(help_text)
        print(help_text)
    
    def stop(self):
        """Stop the application."""
        logger.info("Stopping application...")
        self.state.is_running = False
        self.shutdown_event.set()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Stop components
        if hasattr(self, 'gesture_processor'):
            self.gesture_processor.stop()
        
        if hasattr(self, 'input_handler'):
            self.input_handler.stop()
        
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop()
        
        if hasattr(self, 'config_manager'):
            self.config_manager.stop_hot_reload()
        
        if hasattr(self, 'voice_handler') and self.voice_handler:
            self.voice_handler.stop()
        
        if hasattr(self, 'crash_handler'):
            self.crash_handler.unregister()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hand to Key - Advanced Hand Gesture Control System"
    )
    
    # Camera options
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--auto-camera', action='store_true',
                       help='Auto-select best camera')
    
    # Configuration options
    parser.add_argument('--config-dir', type=str, default='./config',
                       help='Configuration directory')
    parser.add_argument('--profile', type=str,
                       help='Profile to load')
    parser.add_argument('--hot-reload', action='store_true',
                       help='Enable configuration hot-reload')
    
    # Performance options
    parser.add_argument('--threading', action='store_true',
                       help='Enable multi-threading')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--adaptive', action='store_true',
                       help='Enable adaptive quality')
    
    # Feature options
    parser.add_argument('--custom-gestures', action='store_true',
                       help='Enable custom gesture detection')
    parser.add_argument('--model-path', type=str,
                       help='Path to custom gesture model')
    parser.add_argument('--voice-commands', action='store_true',
                       help='Enable voice commands')
    parser.add_argument('--system-tray', action='store_true',
                       help='Run in system tray')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run application
    app = HandToKeyApplication(args)
    
    try:
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
