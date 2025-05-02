import io
import base64
import cv2
import PIL.Image
import mss
from abc import ABC, abstractmethod
import asyncio



class FrameSource(ABC):
    """
    Abstract base class for different frame sources (camera, screen, etc.)
    """
    def __init__(self):
        # The most recent frame captured, to be used for analysis
        self.current_frame = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the frame source"""
        pass
    
    @abstractmethod
    async def get_frame(self):
        """Get a frame from the source, returns formatted data for the API"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass
    
    def get_current_frame_for_analysis(self):
        """Return the current frame for analysis"""
        return self.current_frame


class CameraFrameSource(FrameSource):
    """
    Camera-based frame source that captures frames from a webcam
    """
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.cap = None
    
    async def initialize(self):
        """Initialize the camera capture"""
        self.cap = await asyncio.to_thread(cv2.VideoCapture, self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
    
    def _capture_frame(self):
        """Capture a frame from the camera (internal method)"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB color space (OpenCV captures in BGR, PIL expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])  # Resize to reasonable dimensions
        
        # Store frame for analysis
        self.current_frame = img.copy()
        
        # Convert to format required by Gemini API
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
    
    async def get_frame(self):
        """Get a frame from the camera"""
        return await asyncio.to_thread(self._capture_frame)
    
    async def cleanup(self):
        """Release the camera resources"""
        if self.cap:
            self.cap.release()


class ScreenFrameSource(FrameSource):
    """
    Screen-based frame source that captures the computer screen
    """
    def __init__(self, monitor_index=0):
        super().__init__()
        self.monitor_index = monitor_index
    
    async def initialize(self):
        """Initialize screen capture (no initialization needed)"""
        pass
    
    def _capture_screen(self):
        """Capture a screenshot (internal method)"""
        sct = mss.mss()
        monitor = sct.monitors[self.monitor_index]
        
        screenshot = sct.grab(monitor)
        
        image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))
        
        # Store frame for analysis
        self.current_frame = img.copy()
        
        # Convert to format required by Gemini API
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
    
    async def get_frame(self):
        """Get a screenshot of the screen"""
        return await asyncio.to_thread(self._capture_screen)
    
    async def cleanup(self):
        """Clean up resources (nothing to clean up for screen capture)"""
        pass

