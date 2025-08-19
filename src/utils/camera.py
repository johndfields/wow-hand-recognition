"""
Camera management utilities with auto-selection and adaptive quality.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Information about a camera device."""
    index: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    is_available: bool
    score: float = 0.0


class CameraSelector:
    """Automatically selects the best available camera."""
    
    def __init__(self):
        self.cameras: List[CameraInfo] = []
    
    def scan_cameras(self, max_index: int = 10) -> List[CameraInfo]:
        """Scan for available cameras."""
        cameras = []
        
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to get camera name (platform-specific)
                name = f"Camera {i}"
                
                camera_info = CameraInfo(
                    index=i,
                    name=name,
                    resolution=(width, height),
                    fps=fps,
                    is_available=True
                )
                
                # Calculate score based on resolution and FPS
                camera_info.score = self._calculate_camera_score(camera_info)
                cameras.append(camera_info)
                
                cap.release()
                logger.info(f"Found camera {i}: {width}x{height} @ {fps}fps (score: {camera_info.score:.2f})")
        
        self.cameras = cameras
        return cameras
    
    def _calculate_camera_score(self, camera: CameraInfo) -> float:
        """Calculate a score for camera quality."""
        width, height = camera.resolution
        
        # Prefer HD resolution
        resolution_score = min(width * height / (1920 * 1080), 1.0) * 50
        
        # Prefer 30+ FPS
        fps_score = min(camera.fps / 30, 1.0) * 30
        
        # Prefer lower index (usually built-in camera)
        index_score = max(20 - camera.index * 2, 0)
        
        return resolution_score + fps_score + index_score
    
    def select_best_camera(self) -> int:
        """Select the best available camera."""
        if not self.cameras:
            self.scan_cameras()
        
        if not self.cameras:
            logger.warning("No cameras found, using default index 0")
            return 0
        
        # Sort by score
        sorted_cameras = sorted(self.cameras, key=lambda c: c.score, reverse=True)
        best_camera = sorted_cameras[0]
        
        logger.info(f"Selected camera {best_camera.index} with score {best_camera.score:.2f}")
        return best_camera.index


class CameraManager:
    """Manages camera capture with adaptive quality and performance optimization."""
    
    def __init__(self, camera_index: int = 0, 
                 width: int = 960, 
                 height: int = 540,
                 fps: int = 30,
                 adaptive_quality: bool = True,
                 buffer_size: int = 1):
        
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.adaptive_quality = adaptive_quality
        self.buffer_size = buffer_size
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_resolution = (width, height)
        self.current_fps = fps
        
        # Performance tracking
        self.frame_times: List[float] = []
        self.last_frame_time = time.time()
        self.fps_history: List[float] = []
        
        # Threading for background capture
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_buffer: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Adaptive quality parameters
        self.quality_level = 2  # 0: Low, 1: Medium, 2: High
        self.quality_resolutions = [
            (640, 360),   # Low
            (960, 540),   # Medium
            (1280, 720)   # High
        ]
    
    def start(self) -> bool:
        """Start camera capture."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set initial resolution
            self._set_resolution(self.target_width, self.target_height)
            
            # Set FPS
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Verify settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.current_resolution = (actual_width, actual_height)
            self.current_fps = actual_fps
            self.is_running = True
            
            # Start capture thread if enabled
            if self.buffer_size > 0:
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info("Camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from camera."""
        if not self.is_running or not self.capture:
            return None
        
        try:
            if self.buffer_size > 0 and self.frame_buffer is not None:
                # Get from buffer
                with self.frame_lock:
                    frame = self.frame_buffer.copy()
            else:
                # Direct capture
                ret, frame = self.capture.read()
                if not ret:
                    return None
            
            # Update FPS tracking
            self._update_fps()
            
            # Adaptive quality adjustment
            if self.adaptive_quality:
                self._adjust_quality()
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def _capture_loop(self):
        """Background thread for continuous capture."""
        while self.is_running:
            try:
                ret, frame = self.capture.read()
                if ret:
                    with self.frame_lock:
                        self.frame_buffer = frame
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.01)
    
    def _set_resolution(self, width: int, height: int):
        """Set camera resolution."""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.current_resolution = (width, height)
    
    def _update_fps(self):
        """Update FPS tracking."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if len(self.frame_times) >= 10:
            avg_frame_time = np.mean(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.fps_history.append(current_fps)
            
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
    
    def _adjust_quality(self):
        """Adjust quality based on performance."""
        if len(self.fps_history) < 10:  # Wait for more data before adjusting
            return
        
        avg_fps = np.mean(self.fps_history)
        target_fps_ratio = avg_fps / self.target_fps
        
        # More conservative thresholds to reduce frequent changes
        if target_fps_ratio < 0.6 and self.quality_level > 0:  # Lower threshold
            # Only decrease if consistently poor performance
            recent_fps = np.mean(self.fps_history[-5:])
            if recent_fps / self.target_fps < 0.65:
                self.quality_level -= 1
                new_resolution = self.quality_resolutions[self.quality_level]
                self._set_resolution(*new_resolution)
                logger.info(f"Decreased quality to level {self.quality_level}: {new_resolution}")
                self.fps_history.clear()
            
        elif target_fps_ratio > 1.1 and self.quality_level < 2:  # Higher threshold
            # Only increase if consistently good performance and current level stable
            recent_fps = np.mean(self.fps_history[-5:])
            if recent_fps / self.target_fps > 1.05:
                self.quality_level += 1
                new_resolution = self.quality_resolutions[self.quality_level]
                self._set_resolution(*new_resolution)
                logger.info(f"Increased quality to level {self.quality_level}: {new_resolution}")
                self.fps_history.clear()
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if self.fps_history:
            return np.mean(self.fps_history)
        return 0.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        return self.current_resolution
    
    def set_brightness(self, value: float):
        """Set camera brightness (0.0 to 1.0)."""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, value)
    
    def set_contrast(self, value: float):
        """Set camera contrast (0.0 to 1.0)."""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_CONTRAST, value)
    
    def set_exposure(self, value: float):
        """Set camera exposure."""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
    
    def auto_adjust_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Automatically adjust frame for better lighting conditions."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return adjusted
    
    def apply_background_subtraction(self, frame: np.ndarray, 
                                   background_subtractor=None) -> np.ndarray:
        """Apply background subtraction for better hand detection."""
        if background_subtractor is None:
            background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True
            )
        
        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        return result
