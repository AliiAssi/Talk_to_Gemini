import asyncio
import base64
import io
import traceback
import time

import cv2
import pyaudio
import PIL.Image
import mss
import argparse

from google import genai
from google.genai import types

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Audio parameters (this will be in the frontend)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Model and modes
MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "screen"
DEFAULT_SUGGESTION_INTERVAL = 20

# Initialize client
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=API_KEY
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

pya = pyaudio.PyAudio()

class ScreenAnalyzer:
    def __init__(self, video_mode=DEFAULT_MODE, suggestion_interval=DEFAULT_SUGGESTION_INTERVAL):
        self.video_mode = video_mode
        self.suggestion_interval = suggestion_interval
        self.last_suggestion_time = 0
        self.frame_for_analysis = None
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.context_history = []  # Store previous context for smarter recommendations
        self.audio_stream = None

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        img = PIL.Image.open(io.BytesIO(mss.tools.to_png(shot.rgb, shot.size)))
        img.thumbnail([1024, 1024])
        self.frame_for_analysis = img.copy()
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        return {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break
            await asyncio.sleep(0.5)  # Reduced sleep time for more responsive analysis
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            data = msg["data"]
            raw = base64.b64decode(data) if isinstance(data, str) else data
            blob = types.Blob(data=raw, mime_type=msg["mime_type"])
            await self.session.send_realtime_input(media=blob)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info['index'],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {'exception_on_overflow': False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({'data': data, 'mime_type': 'audio/pcm'})

    async def receive_audio(self):
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                elif text := response.text:
                    print(text, end="")
            # clear audio queue on interrupt
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            chunk = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, chunk)

    async def smart_analysis(self):
        """Proactively analyze screen content and provide intelligent suggestions."""
        # Initial system prompt to set the analyzer's behavior
        system_prompt = """You are an intelligent screen analyzer and game assistant that provides helpful, 
        contextual suggestions based on what you see on the user's screen.
        
        Focus on:
        1. Identifying what the user is doing.
        2. Suggesting useful shortcuts, tips, or improvements.
        3. Being concise and non-intrusive.
        4. Minimal few simple words.
        4. Adapting to context - only mention significant changes.
        5. Being helpful without being annoying.
        
        Keep suggestions brief and actionable."""
        
        # Send initial system prompt once
        await self.session.send_realtime_input(text=system_prompt)
        await asyncio.sleep(2)
        
        
        # Start periodic analysis
        while True:
            now = time.time()
            if now - self.last_suggestion_time >= self.suggestion_interval and self.frame_for_analysis:
                buf = io.BytesIO()
                self.frame_for_analysis.save(buf, format="jpeg")
                raw = buf.getvalue()
                blob = types.Blob(data=raw, mime_type="image/jpeg")
                
                # Create a context-aware prompt
                prompt = """Analyze this screen and provide one brief, helpful suggestion.
                Be specific about what you see and keep your response under 3 sentences.
                Only mention significant changes or opportunities for improvement."""
                
                print("\n[Analyzing screen content...]")
                
                # Send the analysis prompt
                await self.session.send_realtime_input(text=prompt)
                
                # Send the current frame for analysis
                await self.session.send_realtime_input(media=blob)
                
                self.last_suggestion_time = now
            
            await asyncio.sleep(3)  # Check more frequently than the suggestion interval

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                # print(self.session)
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Core tasks for I/O
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio()) # it is the frontend task
                tg.create_task(self.get_screen()) # it is the frontend task
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio()) # it is the frontend task
                
                # Smart analysis task
                tg.create_task(self.smart_analysis())

                # Block until cancelled
                await asyncio.Future()
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(eg)

if __name__ == "__main__":
    analyzer = ScreenAnalyzer(video_mode="screen")
    print(f"Screen Analyzer starting - providing suggestions every 20 seconds")
    asyncio.run(analyzer.run())