"""
Placeholder/stub implementations for utility modules.
These should be implemented separately based on specific requirements.
"""

import logging
from typing import Dict, Any, Callable, Optional, List
import time

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Placeholder for audio feedback system."""
    
    def __init__(self, volume: float = 0.5, enabled: bool = True):
        self.volume = volume
        self.enabled = enabled
        logger.info("AudioFeedback initialized (stub)")
    
    def play_gesture_sound(self, gesture: str):
        """Play sound for gesture."""
        if self.enabled:
            logger.debug(f"Playing sound for gesture: {gesture}")
    
    def play_system_sound(self, sound_type: str):
        """Play system sound."""
        if self.enabled:
            logger.debug(f"Playing system sound: {sound_type}")
    
    def set_volume(self, volume: float):
        """Set volume."""
        self.volume = volume
    
    def set_enabled(self, enabled: bool):
        """Enable/disable audio."""
        self.enabled = enabled


class StatisticsTracker:
    """Comprehensive statistics tracking with real-time metrics."""
    
    def __init__(self):
        self.total_gestures = 0
        self.successful_gestures = 0
        self.gesture_counts = {}
        self.gesture_success_counts = {}
        
        # Performance tracking
        self.fps_history = []
        self.frame_times = []
        self.processing_times = []
        self.confidence_history = []
        
        # Session tracking
        self.session_start_time = time.time()
        self.last_gesture_time = 0.0
        self.gesture_frequency = 0.0
        
        # Real-time metrics
        self.current_fps = 0.0
        self.average_confidence = 0.0
        self.average_processing_time = 0.0
        
        # Gesture stability tracking
        self.gesture_stability_scores = {}
        self.active_gesture_durations = {}
        
        logger.info("StatisticsTracker initialized with comprehensive metrics")
    
    def record_gesture(self, gesture: str, success: bool, confidence: float = 1.0):
        """Record a gesture with confidence score."""
        current_time = time.time()
        
        self.total_gestures += 1
        if success:
            self.successful_gestures += 1
            
            if gesture not in self.gesture_success_counts:
                self.gesture_success_counts[gesture] = 0
            self.gesture_success_counts[gesture] += 1
        
        if gesture not in self.gesture_counts:
            self.gesture_counts[gesture] = 0
        self.gesture_counts[gesture] += 1
        
        # Update timing metrics
        if self.last_gesture_time > 0:
            gesture_interval = current_time - self.last_gesture_time
            self.gesture_frequency = 1.0 / gesture_interval if gesture_interval > 0 else 0.0
        
        self.last_gesture_time = current_time
        
        # Track confidence
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        
        self.average_confidence = sum(self.confidence_history) / len(self.confidence_history)
    
    def record_gesture_continuation(self, gesture: str, confidence: float = 1.0):
        """Record gesture continuation (no state change) for cleaner statistics.
        
        This method is called when a gesture is detected but is already active,
        providing a way to track gesture stability and confidence without
        inflating activation statistics.
        """
        # Update confidence tracking for the gesture
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        
        self.average_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Track gesture stability (how consistently it's being detected)
        if gesture not in self.gesture_stability_scores:
            self.gesture_stability_scores[gesture] = []
        
        # Higher confidence during continuation indicates more stability
        self.gesture_stability_scores[gesture].append(confidence)
        if len(self.gesture_stability_scores[gesture]) > 20:
            self.gesture_stability_scores[gesture].pop(0)
        
        # Log continuation for debugging (at debug level to avoid spam)
        logger.debug(f"Gesture continuation: {gesture} (confidence: {confidence:.2f})")
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:  # Keep last 60 frame times
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def record_processing_time(self, processing_time: float):
        """Record gesture processing time."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)
    
    def record_gesture_stability(self, gesture: str, stability_score: float):
        """Record gesture stability score."""
        if gesture not in self.gesture_stability_scores:
            self.gesture_stability_scores[gesture] = []
        
        self.gesture_stability_scores[gesture].append(stability_score)
        if len(self.gesture_stability_scores[gesture]) > 20:
            self.gesture_stability_scores[gesture].pop(0)
    
    def start_gesture_duration(self, gesture: str):
        """Start tracking gesture duration."""
        self.active_gesture_durations[gesture] = time.time()
    
    def end_gesture_duration(self, gesture: str) -> float:
        """End tracking gesture duration and return duration."""
        if gesture in self.active_gesture_durations:
            duration = time.time() - self.active_gesture_durations[gesture]
            del self.active_gesture_durations[gesture]
            return duration
        return 0.0
    
    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_gestures == 0:
            return 0.0
        return self.successful_gestures / self.total_gestures
    
    def get_gesture_success_rate(self, gesture: str) -> float:
        """Get success rate for specific gesture."""
        if gesture not in self.gesture_counts or self.gesture_counts[gesture] == 0:
            return 0.0
        
        successful = self.gesture_success_counts.get(gesture, 0)
        return successful / self.gesture_counts[gesture]
    
    def get_session_duration(self) -> float:
        """Get current session duration in seconds."""
        return time.time() - self.session_start_time
    
    def get_most_used_gesture(self) -> str:
        """Get the most frequently used gesture."""
        if not self.gesture_counts:
            return "None"
        return max(self.gesture_counts.items(), key=lambda x: x[1])[0]
    
    def get_gesture_stability(self, gesture: str) -> float:
        """Get average stability score for gesture."""
        if gesture not in self.gesture_stability_scores:
            return 0.0
        
        scores = self.gesture_stability_scores[gesture]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'fps': self.current_fps,
            'avg_processing_time': self.average_processing_time,
            'avg_confidence': self.average_confidence,
            'gesture_frequency': self.gesture_frequency
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        session_duration = self.get_session_duration()
        
        return {
            # Basic counts
            'total_gestures': self.total_gestures,
            'successful_gestures': self.successful_gestures,
            'success_rate': self.get_success_rate(),
            'gesture_counts': self.gesture_counts.copy(),
            
            # Performance metrics
            'current_fps': self.current_fps,
            'average_confidence': self.average_confidence,
            'average_processing_time': self.average_processing_time,
            'gesture_frequency': self.gesture_frequency,
            
            # Session info
            'session_duration': session_duration,
            'gestures_per_minute': (self.total_gestures / (session_duration / 60)) if session_duration > 0 else 0.0,
            'most_used_gesture': self.get_most_used_gesture(),
            
            # Stability metrics
            'gesture_stability_scores': {k: self.get_gesture_stability(k) for k in self.gesture_counts.keys()}
        }
    
    def update(self):
        """Update real-time statistics."""
        # Update FPS history
        self.fps_history.append(self.current_fps)
        if len(self.fps_history) > 300:  # Keep 5 minutes at 60fps
            self.fps_history.pop(0)
    
    def reset_session(self):
        """Reset session statistics."""
        self.total_gestures = 0
        self.successful_gestures = 0
        self.gesture_counts.clear()
        self.gesture_success_counts.clear()
        self.session_start_time = time.time()
        self.last_gesture_time = 0.0
        logger.info("Session statistics reset")


class CrashHandler:
    """Placeholder for crash handling and recovery."""
    
    def __init__(self, app_restart_callback: Callable = None,
                 state_save_callback: Callable = None,
                 state_load_callback: Callable = None):
        self.app_restart_callback = app_restart_callback
        self.state_save_callback = state_save_callback
        self.state_load_callback = state_load_callback
        logger.info("CrashHandler initialized (stub)")
    
    def register(self):
        """Register crash handler."""
        logger.debug("Crash handler registered")
    
    def unregister(self):
        """Unregister crash handler."""
        logger.debug("Crash handler unregistered")
    
    def handle_crash(self, exception: Exception):
        """Handle a crash."""
        logger.error(f"Handling crash: {exception}")
        if self.state_save_callback:
            state = self.state_save_callback()
            logger.info(f"State saved: {state}")


class OverlayRenderer:
    """Comprehensive overlay rendering with themes, statistics, and multiple display modes."""
    
    def __init__(self, show_statistics: bool = True,
                 show_debug_info: bool = False,
                 ui_scale: float = 1.0,
                 theme: str = "dark",
                 config_manager=None):
        self.show_statistics = show_statistics
        self.show_debug_info = show_debug_info
        self.ui_scale = ui_scale
        self.theme = theme
        self.high_contrast = False
        self.text_scale = 1.0
        self.config_manager = config_manager
        
        # Display modes
        self.display_mode = "normal"  # "normal", "minimal", "detailed", "debug"
        self.show_keybindings = True
        self.show_performance = True
        self.show_gesture_history = True
        
        # UI state
        self.last_active_gestures = set()
        self.gesture_feedback_timer = {}
        self.notification_queue = []
        
        # Color schemes
        self._init_color_schemes()
        
        logger.info(f"OverlayRenderer initialized with theme: {theme}, scale: {ui_scale}")
    
    def _init_color_schemes(self):
        """Initialize color schemes for different themes."""
        import cv2
        
        self.color_schemes = {
            "dark": {
                "background": (30, 30, 30),
                "text": (255, 255, 255),
                "accent": (0, 255, 255),
                "success": (0, 255, 0),
                "warning": (0, 255, 255),
                "error": (0, 0, 255),
                "inactive": (128, 128, 128),
                "active_gesture": (0, 255, 0),
                "panel_bg": (20, 20, 20)
            },
            "light": {
                "background": (240, 240, 240),
                "text": (20, 20, 20),
                "accent": (200, 100, 0),
                "success": (0, 150, 0),
                "warning": (200, 150, 0),
                "error": (200, 0, 0),
                "inactive": (120, 120, 120),
                "active_gesture": (0, 180, 0),
                "panel_bg": (220, 220, 220)
            },
            "high_contrast": {
                "background": (0, 0, 0),
                "text": (255, 255, 255),
                "accent": (255, 255, 0),
                "success": (0, 255, 0),
                "warning": (255, 255, 0),
                "error": (255, 0, 0),
                "inactive": (128, 128, 128),
                "active_gesture": (0, 255, 0),
                "panel_bg": (0, 0, 0)
            }
        }
    
    def get_colors(self) -> Dict[str, tuple]:
        """Get current color scheme."""
        if self.high_contrast:
            return self.color_schemes["high_contrast"]
        return self.color_schemes.get(self.theme, self.color_schemes["dark"])
    
    def render(self, frame, detections=None, stats=None, state=None):
        """Render comprehensive overlay on frame."""
        import cv2
        import numpy as np
        
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        colors = self.get_colors()
        
        # Create overlay based on display mode
        if self.display_mode == "minimal":
            return self._render_minimal(frame, detections, stats, state, colors)
        elif self.display_mode == "detailed":
            return self._render_detailed(frame, detections, stats, state, colors)
        elif self.display_mode == "debug":
            return self._render_debug(frame, detections, stats, state, colors)
        else:
            return self._render_normal(frame, detections, stats, state, colors)
    
    def _render_normal(self, frame, detections, stats, state, colors):
        """Render normal overlay mode."""
        import cv2
        h, w = frame.shape[:2]
        
        # Status bar at top
        self._draw_status_bar(frame, state, colors)
        
        # Keybindings panel on the right
        if self.show_keybindings:
            self._draw_keybindings_panel(frame, state, colors)
        
        # Active gestures display
        self._draw_active_gestures(frame, detections, colors)
        
        # Performance metrics
        if self.show_performance and self.show_statistics and stats:
            self._draw_performance_metrics(frame, stats, colors)
        
        # Instructions at bottom
        self._draw_instructions(frame, colors)
        
        # Notifications
        self._draw_notifications(frame, colors)
        
        return frame
    
    def _render_minimal(self, frame, detections, stats, state, colors):
        """Render minimal overlay mode."""
        import cv2
        h, w = frame.shape[:2]
        
        # Just status and active gestures
        status = "PAUSED" if (state and state.is_paused) else "ACTIVE"
        color = colors["error"] if (state and state.is_paused) else colors["success"]
        
        # Status indicator (small)
        cv2.circle(frame, (20, 20), int(8 * self.ui_scale), color, -1)
        
        # Active gestures (compact)
        if detections:
            y_offset = 50
            for detection in detections[:3]:  # Show max 3
                gesture_name = self._format_gesture_name(detection.gesture_type.value)
                cv2.putText(frame, gesture_name, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale,
                           colors["active_gesture"], 1)
                y_offset += 20
        
        return frame
    
    def _render_detailed(self, frame, detections, stats, state, colors):
        """Render detailed overlay mode."""
        # Include everything from normal mode plus extra details
        frame = self._render_normal(frame, detections, stats, state, colors)
        
        # Additional detailed information
        self._draw_gesture_confidence(frame, detections, colors)
        self._draw_stability_indicators(frame, detections, colors)
        
        if stats:
            self._draw_detailed_statistics(frame, stats, colors)
        
        return frame
    
    def _render_debug(self, frame, detections, stats, state, colors):
        """Render debug overlay mode."""
        import cv2
        
        # Start with detailed mode
        frame = self._render_detailed(frame, detections, stats, state, colors)
        
        # Add debug information
        if detections:
            self._draw_debug_info(frame, detections, colors)
        
        # Frame information
        h, w = frame.shape[:2]
        debug_info = [
            f"Frame: {w}x{h}",
            f"Scale: {self.ui_scale:.1f}",
            f"Theme: {self.theme}",
            f"Mode: {self.display_mode}"
        ]
        
        y_offset = h - 100
        for info in debug_info:
            cv2.putText(frame, info, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale,
                       colors["text"], 1)
            y_offset += 15
        
        return frame
    
    def _draw_status_bar(self, frame, state, colors):
        """Draw status bar at top of frame."""
        import cv2
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(40 * self.ui_scale)), colors["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status = "PAUSED" if (state and state.is_paused) else "ACTIVE"
        color = colors["error"] if (state and state.is_paused) else colors["success"]
        
        font_scale = 0.7 * self.text_scale * self.ui_scale
        cv2.putText(frame, f"Status: {status}", (int(10 * self.ui_scale), int(25 * self.ui_scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        # Profile information
        if state and hasattr(state, 'current_profile') and state.current_profile:
            profile_text = f"Profile: {state.current_profile}"
            cv2.putText(frame, profile_text, (int(200 * self.ui_scale), int(25 * self.ui_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, colors["accent"], 1)
    
    def _draw_keybindings_panel(self, frame, state, colors):
        """Draw keybindings panel on the right side."""
        import cv2
        h, w = frame.shape[:2]
        
        panel_width = int(280 * self.ui_scale)
        panel_x = w - panel_width
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, int(50 * self.ui_scale)), 
                     (w, h - int(50 * self.ui_scale)), colors["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        font_scale = 0.6 * self.text_scale * self.ui_scale
        cv2.putText(frame, "Keybindings:", (panel_x + int(10 * self.ui_scale), int(75 * self.ui_scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors["accent"], 2)
        
        # Get gesture mappings from state
        gesture_mappings = self._get_gesture_mappings(state)
        
        y_offset = int(100 * self.ui_scale)
        for gesture, mapping in gesture_mappings.items():
            display_name = self._format_gesture_name(gesture)
            target = mapping.get('target', 'N/A').upper()
            mode = mapping.get('mode', 'tap').upper()
            
            # Highlight active gestures
            text_color = colors["active_gesture"] if gesture in self.last_active_gestures else colors["text"]
            
            text = f"{display_name}: {target} ({mode})"
            cv2.putText(frame, text, (panel_x + int(10 * self.ui_scale), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale * self.ui_scale, text_color, 1)
            y_offset += int(22 * self.ui_scale)
    
    def _draw_active_gestures(self, frame, detections, colors):
        """Draw currently active gestures."""
        import cv2
        h, w = frame.shape[:2]
        
        if not detections:
            return
        
        # Update active gestures
        current_gestures = {d.gesture_type.value for d in detections}
        self.last_active_gestures = current_gestures
        
        # Active gestures panel (bottom left)
        y_start = h - int(120 * self.ui_scale)
        
        if current_gestures:
            cv2.putText(frame, "Active Gestures:", (int(10 * self.ui_scale), y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.text_scale * self.ui_scale, 
                       colors["success"], 2)
            
            y_offset = y_start + int(25 * self.ui_scale)
            for detection in detections[:4]:  # Show max 4
                gesture_name = self._format_gesture_name(detection.gesture_type.value)
                confidence_text = f" ({detection.confidence:.2f})" if hasattr(detection, 'confidence') else ""
                text = f"• {gesture_name}{confidence_text}"
                
                cv2.putText(frame, text, (int(15 * self.ui_scale), y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale * self.ui_scale,
                           colors["active_gesture"], 1)
                y_offset += int(20 * self.ui_scale)
    
    def _draw_performance_metrics(self, frame, stats, colors):
        """Draw performance metrics."""
        import cv2
        h, w = frame.shape[:2]
        
        # Performance panel (top right, below keybindings title)
        panel_x = w - int(280 * self.ui_scale)
        y_start = int(110 * self.ui_scale)
        
        metrics = [
            ("FPS", f"{stats.get('fps', 0):.1f}"),
            ("Confidence", f"{stats.get('avg_confidence', 0):.2f}"),
            ("Gestures", str(stats.get('gestures_detected', 0)))
        ]
        
        y_offset = y_start
        for label, value in metrics:
            text = f"{label}: {value}"
            cv2.putText(frame, text, (panel_x + int(10 * self.ui_scale), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale * self.ui_scale,
                       colors["text"], 1)
            y_offset += int(18 * self.ui_scale)
    
    def _draw_instructions(self, frame, colors):
        """Draw instructions at bottom."""
        import cv2
        h, w = frame.shape[:2]
        
        instructions = [
            "q: Quit",
            "p: Pause/Resume", 
            "h: Help",
            "1-9: Switch Profile",
            "d: Toggle Debug"
        ]
        
        instruction_text = " | ".join(instructions)
        cv2.putText(frame, instruction_text, (int(10 * self.ui_scale), h - int(10 * self.ui_scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale * self.ui_scale,
                   colors["inactive"], 1)
    
    def _draw_gesture_confidence(self, frame, detections, colors):
        """Draw gesture confidence indicators."""
        import cv2
        
        if not detections:
            return
        
        # Confidence bars on the left side
        x_start = int(10 * self.ui_scale)
        y_start = int(150 * self.ui_scale)
        bar_width = int(100 * self.ui_scale)
        bar_height = int(8 * self.ui_scale)
        
        for i, detection in enumerate(detections[:5]):
            y_pos = y_start + i * int(25 * self.ui_scale)
            
            # Background bar
            cv2.rectangle(frame, (x_start, y_pos), 
                         (x_start + bar_width, y_pos + bar_height),
                         colors["inactive"], -1)
            
            # Confidence bar
            if hasattr(detection, 'confidence'):
                conf_width = int(bar_width * detection.confidence)
                color = colors["success"] if detection.confidence > 0.7 else colors["warning"]
                cv2.rectangle(frame, (x_start, y_pos),
                             (x_start + conf_width, y_pos + bar_height),
                             color, -1)
                
                # Confidence text
                conf_text = f"{detection.confidence:.2f}"
                cv2.putText(frame, conf_text, (x_start + bar_width + int(5 * self.ui_scale), y_pos + bar_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3 * self.text_scale * self.ui_scale,
                           colors["text"], 1)
    
    def _draw_stability_indicators(self, frame, detections, colors):
        """Draw gesture stability indicators."""
        # This would show temporal filtering status
        # Implementation depends on stability data from gesture processor
        pass
    
    def _draw_detailed_statistics(self, frame, stats, colors):
        """Draw detailed statistics panel."""
        import cv2
        h, w = frame.shape[:2]
        
        # Detailed stats panel (left side, middle)
        x_start = int(10 * self.ui_scale)
        y_start = int(300 * self.ui_scale)
        
        detailed_stats = [
            ("Session", f"{stats.get('session_duration', 0) / 60:.1f}m"),
            ("Success Rate", f"{stats.get('success_rate', 0) * 100:.1f}%"),
            ("Most Used", stats.get('most_used_gesture', 'None')[:10])
        ]
        
        y_offset = y_start
        for label, value in detailed_stats:
            text = f"{label}: {value}"
            cv2.putText(frame, text, (x_start, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale * self.ui_scale,
                       colors["text"], 1)
            y_offset += int(20 * self.ui_scale)
    
    def _draw_debug_info(self, frame, detections, colors):
        """Draw debug information for gestures."""
        import cv2
        
        if not detections:
            return
        
        # Debug info for each detection
        for i, detection in enumerate(detections):
            if hasattr(detection, 'landmarks') and detection.landmarks:
                # Draw hand landmarks if available
                # This would integrate with MediaPipe drawing utilities
                pass
    
    def _draw_notifications(self, frame, colors):
        """Draw notification messages."""
        import cv2
        import time
        h, w = frame.shape[:2]
        
        # Clean up expired notifications
        current_time = time.time()
        self.notification_queue = [(msg, expire_time) for msg, expire_time in self.notification_queue 
                                  if expire_time > current_time]
        
        # Draw active notifications
        y_offset = int(60 * self.ui_scale)
        for msg, _ in self.notification_queue[:3]:  # Show max 3 notifications
            # Semi-transparent background
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5 * self.text_scale * self.ui_scale, 1)[0]
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(w/2 - text_size[0]/2 - 10), y_offset - int(20 * self.ui_scale)),
                         (int(w/2 + text_size[0]/2 + 10), y_offset + int(5 * self.ui_scale)),
                         colors["panel_bg"], -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Notification text
            cv2.putText(frame, msg, (int(w/2 - text_size[0]/2), y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale * self.ui_scale,
                       colors["accent"], 1)
            y_offset += int(30 * self.ui_scale)
    
    def _get_gesture_mappings(self, state) -> Dict[str, Dict[str, str]]:
        """Get gesture mappings from application state."""
        if not self.config_manager or not self.config_manager.active_profile:
            # Fallback to default mappings if no config manager or active profile
            return {
                "swipe_left": {"target": "alt+left", "mode": "tap"},
                "swipe_right": {"target": "alt+right", "mode": "tap"},
                "swipe_up": {"target": "page_up", "mode": "tap"},
                "swipe_down": {"target": "page_down", "mode": "tap"},
                "pinch_index": {"target": "ctrl+c", "mode": "tap"},
                "pinch_middle": {"target": "ctrl+v", "mode": "tap"},
                "pinch_ring": {"target": "ctrl+z", "mode": "tap"},
                "pinch_pinky": {"target": "ctrl+y", "mode": "tap"},
                "open_palm": {"target": "ctrl+s", "mode": "tap"},
                "fist": {"target": "escape", "mode": "tap"}
            }
        
        # Get gesture mappings from active profile
        mappings = {}
        for gesture_mapping in self.config_manager.active_profile.gesture_mappings:
            if gesture_mapping.enabled:
                mappings[gesture_mapping.gesture] = {
                    "target": gesture_mapping.target,
                    "mode": gesture_mapping.mode,
                    "action_type": gesture_mapping.action_type
                }
        
        return mappings
    
    def _format_gesture_name(self, gesture: str) -> str:
        """Format gesture name for display."""
        display_names = {
            'open_palm': 'Open Palm',
            'fist': 'Fist',
            'swipe_left': 'Swipe Left',
            'swipe_right': 'Swipe Right', 
            'swipe_up': 'Swipe Up',
            'swipe_down': 'Swipe Down',
            'pinch_index': 'Pinch Index',
            'pinch_middle': 'Pinch Middle',
            'pinch_ring': 'Pinch Ring',
            'pinch_pinky': 'Pinch Pinky',
            'victory': 'Victory',
            'thumbs_up': 'Thumbs Up',
            'three': 'Three Fingers',
            'index_only': 'Index Only',
            'l_shape': 'L Shape',
            'hang_loose': 'Hang Loose',
            'rock_on': 'Rock On'
        }
        return display_names.get(gesture, gesture.replace('_', ' ').title())
    
    def add_notification(self, message: str, duration: float = 3.0):
        """Add a notification message."""
        import time
        expire_time = time.time() + duration
        self.notification_queue.append((message, expire_time))
    
    def set_display_mode(self, mode: str):
        """Set display mode."""
        if mode in ["normal", "minimal", "detailed", "debug"]:
            self.display_mode = mode
            logger.info(f"Display mode set to: {mode}")
    
    def toggle_display_mode(self):
        """Toggle between display modes."""
        modes = ["normal", "minimal", "detailed", "debug"]
        current_idx = modes.index(self.display_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.set_display_mode(modes[next_idx])
    
    def enable_high_contrast(self):
        """Enable high contrast mode."""
        self.high_contrast = True
        logger.info("High contrast mode enabled")
    
    def set_text_scale(self, scale: float):
        """Set text scale."""
        self.text_scale = scale
        logger.info(f"Text scale set to: {scale}")
    
    def set_theme(self, theme: str):
        """Set color theme."""
        if theme in self.color_schemes:
            self.theme = theme
            logger.info(f"Theme set to: {theme}")


class CalibrationMode:
    """Placeholder for calibration mode."""
    
    def __init__(self, gesture_processor=None, overlay_renderer=None):
        self.gesture_processor = gesture_processor
        self.overlay_renderer = overlay_renderer
        self.is_active = False
        self.recorded_gestures = []
        logger.info("CalibrationMode initialized (stub)")
    
    def start(self):
        """Start calibration."""
        self.is_active = True
        self.recorded_gestures = []
        logger.info("Calibration started")
    
    def stop(self) -> Dict[str, Any]:
        """Stop calibration and return results."""
        self.is_active = False
        results = {
            'sensitivity': 1.0,
            'gestures_recorded': len(self.recorded_gestures)
        }
        logger.info(f"Calibration stopped with results: {results}")
        return results
    
    def record_gesture(self, detection):
        """Record a gesture during calibration."""
        self.recorded_gestures.append(detection)
    
    def render(self, frame):
        """Render calibration overlay."""
        return frame


class VoiceCommandHandler:
    """Placeholder for voice command handling."""
    
    def __init__(self, command_callback: Callable = None):
        self.command_callback = command_callback
        self.is_running = False
        logger.info("VoiceCommandHandler initialized (stub)")
    
    def start(self):
        """Start voice command listener."""
        self.is_running = True
        logger.info("Voice command handler started")
    
    def stop(self):
        """Stop voice command listener."""
        self.is_running = False
        logger.info("Voice command handler stopped")


class SystemTrayApp:
    """Placeholder for system tray application."""
    
    def __init__(self, app_controller=None, profiles: List[str] = None):
        self.app_controller = app_controller
        self.profiles = profiles or []
        logger.info("SystemTrayApp initialized (stub)")
    
    def run(self):
        """Run system tray app."""
        logger.info("System tray app running (stub)")
        # In real implementation, this would create a system tray icon
        # and handle menu interactions
