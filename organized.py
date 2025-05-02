import asyncio
import base64
import io
import traceback
import time
from abc import ABC, abstractmethod

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


api_key = os.getenv("MODEL_API_KEY")
# print(api_key[:4])
# Audio configuration constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024  # Size of audio chunks for processing

# Gemini API configuration
MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "screen"

# Default time interval (in seconds) between proactive suggestions
DEFAULT_SUGGESTION_INTERVAL = 30

# Initialize the Gemini API client
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key  
)

# Gemini API configuration for audio responses
CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "audio",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

# Initialize PyAudio instance
pya = pyaudio.PyAudio()


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


class AudioHandler:
    """
    Handles audio input and output
    """
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.audio_stream = None
        self.session = None
    
    async def setup(self, session, audio_in_queue, out_queue):
        """Set up the audio handler"""
        self.session = session
        self.audio_in_queue = audio_in_queue
        self.out_queue = out_queue
    
    async def listen_audio(self):
        """Capture audio from microphone and add to the output queue"""
        # Get default microphone
        mic_info = pya.get_default_input_device_info()
        
        # Initialize audio stream
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        # Handle audio overflow settings based on debug mode
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
            
        # Continuously read audio and add to queue
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
    
    async def receive_audio(self):
        """Receive audio from Gemini API and process it"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                # Handle audio data
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                # Handle text responses
                if text := response.text:
                    print(text, end="")

            # Handle interruptions by clearing the audio queue
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
    
    async def play_audio(self):
        """Play received audio through speakers"""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio_stream:
            self.audio_stream.close()


class FrameAnalyzer:
    """
    Analyzes frames and provides proactive suggestions
    """
    def __init__(self, suggestion_interval=DEFAULT_SUGGESTION_INTERVAL):
        self.suggestion_interval = suggestion_interval
        self.last_suggestion_time = 0
        self.session = None
        self.frame_source = None
    
    async def setup(self, session, frame_source):
        """Set up the frame analyzer"""
        self.session = session
        self.frame_source = frame_source
    
    async def periodic_suggestions(self):
        """Periodically analyze frames and provide proactive suggestions"""
        while True:
            current_time = time.time()
            
            # Check if it's time for a suggestion
            if (current_time - self.last_suggestion_time >= self.suggestion_interval and 
                self.frame_source.get_current_frame_for_analysis() is not None):
                
                # Create a prompt for frame analysis
                prompt = "Based on what you see in this frame, can you provide a helpful suggestion or observation? Be concise and natural."
                
                # Get the current frame for analysis
                current_frame = self.frame_source.get_current_frame_for_analysis()
                
                # Prepare the frame for sending
                image_io = io.BytesIO()
                current_frame.save(image_io, format="jpeg")
                image_io.seek(0)
                image_bytes = image_io.read()
                frame_data = {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}
                
                # Send the frame with the proactive prompt
                print("\n[Proactive suggestion incoming...]")
                await self.session.send(input=frame_data)
                await self.session.send(input=prompt, end_of_turn=True)
                
                # Update the last suggestion time
                self.last_suggestion_time = current_time
            
            # Wait before checking again
            await asyncio.sleep(5)  # Check every 5 seconds


class TextHandler:
    """
    Handles text input from the user
    """
    def __init__(self):
        self.session = None
    
    async def setup(self, session):
        """Set up the text handler"""
        self.session = session
    
    async def send_text(self):
        """Get text input from the user and send to the Gemini API"""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            # Exit if the user types 'q'
            if text.lower() == "q":
                break
            # Send the text to the API
            await self.session.send(input=text or ".", end_of_turn=True)
        
        # Signal completion
        return True


class GeminiLiveApp:
    """
    Main application that coordinates all components
    """
    def __init__(self, video_mode=DEFAULT_MODE, suggestion_interval=DEFAULT_SUGGESTION_INTERVAL):
        # Initialize components
        self.audio_handler = AudioHandler()
        self.frame_analyzer = FrameAnalyzer(suggestion_interval=suggestion_interval)
        self.text_handler = TextHandler()
        
        # Create appropriate frame source based on mode
        if video_mode == "camera":
            self.frame_source = CameraFrameSource()
        elif video_mode == "screen":
            self.frame_source = ScreenFrameSource()
        else:  # "none" mode
            self.frame_source = None
        
        # Initialize queues
        self.audio_in_queue = None
        self.out_queue = None
        
        # Initialize session
        self.session = None
        
        # Store configuration
        self.video_mode = video_mode
    
    async def send_realtime(self):
        """Send frames and audio to the Gemini API"""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)
    
    async def capture_frames(self):
        """Capture frames from the selected source and add to output queue"""
        # Initialize the frame source
        await self.frame_source.initialize()
        
        while True:
            # Get a frame from the source
            frame = await self.frame_source.get_frame()
            if frame is None:
                break
            
            # Add a delay to prevent overwhelming the system
            await asyncio.sleep(1.0)
            
            # Add the frame to the output queue
            await self.out_queue.put(frame)
    
    async def run(self):
        """Run the application"""
        try:
            # Connect to the Gemini API and initialize tasks
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                
                # Initialize queues
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                # Set up components
                await self.audio_handler.setup(session, self.audio_in_queue, self.out_queue)
                await self.text_handler.setup(session)
                
                if self.frame_source:
                    await self.frame_analyzer.setup(session, self.frame_source)
                
                # Create tasks for handling text input
                send_text_task = tg.create_task(self.text_handler.send_text())
                
                # Create task for sending data to the API
                tg.create_task(self.send_realtime())
                
                # Create tasks for handling audio
                tg.create_task(self.audio_handler.listen_audio())
                tg.create_task(self.audio_handler.receive_audio())
                tg.create_task(self.audio_handler.play_audio())
                
                # Create tasks for frame capture and analysis if a frame source is available
                if self.frame_source:
                    tg.create_task(self.capture_frames())
                    tg.create_task(self.frame_analyzer.periodic_suggestions())
                
                # Wait for the text handler to complete (user exited)
                await send_text_task
                
                # Signal exit
                raise asyncio.CancelledError("User requested exit")
        
        except asyncio.CancelledError:
            # Normal exit
            pass
        except ExceptionGroup as EG:
            # Handle errors
            self.audio_handler.cleanup()
            if self.frame_source:
                await self.frame_source.cleanup()
            traceback.print_exception(EG)


def main():
    """Parse command line arguments and run the application"""
    parser = argparse.ArgumentParser(description="Gemini Live API Application")
    
    # Add command line arguments
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Source for visual input (camera, screen, or none)",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--suggestion-interval",
        type=int,
        default=DEFAULT_SUGGESTION_INTERVAL,
        help="Time interval in seconds between proactive suggestions",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and run the application
    app = GeminiLiveApp(video_mode=args.mode, suggestion_interval=args.suggestion_interval)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()