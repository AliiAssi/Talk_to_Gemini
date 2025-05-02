import asyncio
from frames_sources import CameraFrameSource, ScreenFrameSource
from config import (
    FORMAT,
    CHANNELS,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE,
    MODEL,
    DEFAULT_MODE,
    DEFAULT_SUGGESTION_INTERVAL,
)
from audio_handler import AudioHandler
from frame_analyzer import FrameAnalyzer
from text_handler import TextHandler
from google.genai import types
from google import genai
from dotenv import load_dotenv
import os
api_key = os.getenv("MODEL_API_KEY")


client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=api_key  
)

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
            # traceback.print_exception(EG)
