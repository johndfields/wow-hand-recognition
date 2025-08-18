"""
UI preview window for displaying camera feed and debug information.
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Optional, Any, Dict, List

logger = logging.getLogger(__name__)


class PreviewWindow:
    """Preview window for displaying camera feed and debug information."""
    
    def __init__(self, width: int = 960, height: int = 540, window_name: str = "Hand Gesture Recognition",
                 show_stats: bool = True, show_debug: bool = False, scale: float = 1.0):
        """
        Initialize preview window.
        
        Args:
            width: Window width
            height: Window height
            window_name: Name of the window
            show_stats: Whether to show statistics
            show_debug: Whether to show debug information
            scale: UI scale factor
        """
        self.width = width
        self.height = height
        self.window_name = window_name
        self.show_stats = show_stats
        self.show_debug = show_debug
        self.scale = scale
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(width * scale), int(height * scale))
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5 * scale
        self.text_color = (0, 255, 0)  # Green
        self.text_thickness = 1
        self.bg_color = (0, 0, 0)  # Black
        self.bg_alpha = 0.5  # Background opacity
        
        # Gesture visualization
        self.last_gesture = None
        self.gesture_time = 0
        self.gesture_display_time = 2.0  # Display gesture for 2 seconds
        
        logger.info(f"Preview window created: {width}x{height}")
    
    def update(self, frame: Any) -> bool:
        """
        Update the preview window with a new frame.
        
        Args:
            frame: The frame to display
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update performance metrics
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed > 1.0:  # Update FPS every second
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = current_time
            
            # Add statistics if enabled
            if self.show_stats:
                self._add_stats(frame)
            
            # Show the frame
            cv2.imshow(self.window_name, frame)
            
            # Process key events (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            return False
    
    def close(self):
        """Close the preview window."""
        cv2.destroyWindow(self.window_name)
        logger.info("Preview window closed")
    
    def should_exit(self) -> bool:
        """Check if the window should exit (ESC key pressed)."""
        key = cv2.waitKey(1) & 0xFF
        return key == 27  # ESC key
    
    def add_text(self, frame: Any, text: str, position: Tuple[int, int], 
                color: Optional[Tuple[int, int, int]] = None, 
                scale: Optional[float] = None,
                thickness: Optional[int] = None,
                bg: bool = True):
        """
        Add text to the frame.
        
        Args:
            frame: The frame to add text to
            text: The text to add
            position: The position (x, y) to add the text
            color: Text color (B, G, R)
            scale: Font scale
            thickness: Text thickness
            bg: Whether to add a background
        """
        color = color or self.text_color
        scale = scale or self.font_scale
        thickness = thickness or self.text_thickness
        
        # Add background if enabled
        if bg:
            text_size = cv2.getTextSize(text, self.font, scale, thickness)[0]
            bg_rect = (
                position[0], position[1] - text_size[1],
                position[0] + text_size[0], position[1] + 5
            )
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), 
                         self.bg_color, -1)
            cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)
        
        # Add text
        cv2.putText(frame, text, position, self.font, scale, color, thickness)
    
    def _add_stats(self, frame: Any):
        """Add statistics to the frame."""
        # Add FPS
        self.add_text(frame, f"FPS: {self.fps:.1f}", (10, 30))
        
        # Add gesture if recent
        if self.last_gesture and time.time() - self.gesture_time < self.gesture_display_time:
            self.add_text(frame, f"Gesture: {self.last_gesture}", (10, 60), 
                         color=(0, 255, 255))  # Yellow
    
    def set_gesture(self, gesture: str):
        """Set the current detected gesture."""
        self.last_gesture = gesture
        self.gesture_time = time.time()
    
    def set_theme(self, theme: str):
        """Set the UI theme."""
        if theme == "dark":
            self.text_color = (0, 255, 0)  # Green
            self.bg_color = (0, 0, 0)  # Black
        elif theme == "light":
            self.text_color = (0, 0, 0)  # Black
            self.bg_color = (255, 255, 255)  # White
        elif theme == "high_contrast":
            self.text_color = (255, 255, 0)  # Yellow
            self.bg_color = (0, 0, 0)  # Black
    
    def set_scale(self, scale: float):
        """Set the UI scale."""
        self.scale = scale
        self.font_scale = 0.5 * scale
        cv2.resizeWindow(self.window_name, int(self.width * scale), int(self.height * scale))
    
    def add_debug_overlay(self, frame: Any, debug_info: Dict[str, Any]):
        """Add debug information overlay to the frame."""
        if not self.show_debug:
            return
        
        y = 90  # Start below FPS and gesture
        for key, value in debug_info.items():
            self.add_text(frame, f"{key}: {value}", (10, y))
            y += 25
    
    def add_landmarks(self, frame: Any, landmarks: List[Any], connections: List[Tuple[int, int]],
                     landmark_color: Tuple[int, int, int] = (0, 255, 0),
                     connection_color: Tuple[int, int, int] = (255, 0, 0)):
        """
        Add hand landmarks to the frame.
        
        Args:
            frame: The frame to add landmarks to
            landmarks: List of landmark points
            connections: List of connections between landmarks
            landmark_color: Color for landmarks
            connection_color: Color for connections
        """
        h, w, _ = frame.shape
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, start_point, end_point, connection_color, 2)
        
        # Draw landmarks
        for landmark in landmarks:
            point = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, point, 5, landmark_color, -1)

