"""
Camera capture module with enhanced features.
"""

import cv2
import time
import threading
import queue
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CameraCapture:
    """Enhanced camera capture with multi-threading and performance metrics."""
    
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 540, 
                 fps: int = 30, buffer_size: int = 5):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Index of the camera to use
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            buffer_size: Size of the frame buffer
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.is_running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.capture_thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        self.frame_times: List[float] = []
        self.max_frame_times = 30  # Keep track of last 30 frames for FPS calculation
    
    def start(self) -> bool:
        """Start camera capture in a separate thread."""
        if self.is_running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            self.start_time = time.time()
            logger.info(f"Camera {self.camera_index} started at {self.width}x{self.height}, {self.fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera stopped")
    
    def read(self) -> Tuple[bool, Optional[Any]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running:
            return False, None
        
        try:
            # Get frame from buffer
            frame = self.frame_buffer.get(timeout=1.0)
            self.frame_buffer.task_done()
            
            # Update performance metrics
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed > 1.0:  # Update FPS every second
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = current_time
            
            return True, frame
            
        except queue.Empty:
            logger.warning("Frame buffer empty")
            return False, None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def _capture_loop(self):
        """Background thread for capturing frames."""
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("Camera not open in capture loop")
                    time.sleep(0.1)
                    continue
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Add to buffer, dropping oldest frame if full
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.task_done()
                    except queue.Empty:
                        pass
                
                self.frame_buffer.put(frame)
                
                # Track frame time
                current_time = time.time()
                self.frame_times.append(current_time)
                if len(self.frame_times) > self.max_frame_times:
                    self.frame_times.pop(0)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_fps(self) -> float:
        """Get the current FPS."""
        return self.current_fps
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get information about the camera."""
        if not self.cap:
            return {}
        
        return {
            "index": self.camera_index,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.current_fps,
            "backend": self.cap.getBackendName()
        }
    
    def set_property(self, prop_id: int, value: Any) -> bool:
        """Set a camera property."""
        if not self.cap:
            return False
        
        return self.cap.set(prop_id, value)
    
    def get_property(self, prop_id: int) -> Any:
        """Get a camera property."""
        if not self.cap:
            return None
        
        return self.cap.get(prop_id)
    
    def is_available(self) -> bool:
        """Check if the camera is available."""
        if self.cap and self.cap.isOpened():
            return True
        
        # Try to open the camera
        try:
            cap = cv2.VideoCapture(self.camera_index)
            available = cap.isOpened()
            cap.release()
            return available
        except Exception:
            return False

