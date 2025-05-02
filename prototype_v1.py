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

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Model and modes
MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "screen"
DEFAULT_SUGGESTION_INTERVAL = 30  # seconds

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

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, suggestion_interval=DEFAULT_SUGGESTION_INTERVAL):
        self.video_mode = video_mode
        self.suggestion_interval = suggestion_interval
        self.last_suggestion_time = 0
        self.frame_for_analysis = None
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.is_first_frame = True  # Track whether this is the first frame

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        # Convert BGR->RGB, thumbnail, and store
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])
        self.frame_for_analysis = img.copy()
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        return {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)
        cap.release()

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
            await asyncio.sleep(1.0)
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

    async def send_initial_prompt(self):
        """Send an initial text prompt to start the conversation."""
        initial_prompt = "Hello! I'm your AI assistant. I'll analyze what I see on your screen and provide helpful suggestions and insights. Let me take a look at what you're working on."
        await self.session.send_realtime_input(text=initial_prompt)
        print("\n[Sent initial greeting...]")

    async def periodic_suggestions(self):
        """Periodically stream frames with text prompts for analysis and recommendations."""
        # Send an initial prompt right away
        await self.send_initial_prompt()
        
        while True:
            now = time.time()
            if now - self.last_suggestion_time >= self.suggestion_interval and self.frame_for_analysis:
                buf = io.BytesIO()
                self.frame_for_analysis.save(buf, format="jpeg")
                raw = buf.getvalue()
                blob = types.Blob(data=raw, mime_type="image/jpeg")
                
                # Define the prompt based on whether this is the first frame or a subsequent one
                if self.is_first_frame:
                    prompt = "Please analyze what you see on the screen and provide an initial assessment with helpful suggestions."
                    self.is_first_frame = False
                else:
                    prompt = "Based on what you can see now, please provide updated recommendations or insights if anything has changed."
                
                # Send the text prompt first
                await self.session.send_realtime_input(text=prompt)
                print(f"\n[Streaming frame for analysis with prompt: '{prompt}']")
                
                # Then send the frame
                await self.session.send_realtime_input(media=blob)
                self.last_suggestion_time = now
            
            await asyncio.sleep(5)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Tasks for audio/video I/O
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Task for periodic analysis and recommendations
                tg.create_task(self.periodic_suggestions())

                # Block until cancelled
                await asyncio.Future()
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            self.audio_stream.close()
            traceback.print_exception(eg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="pixels to stream from",
    )
    parser.add_argument(
        "--suggestion-interval",
        type=int,
        default=DEFAULT_SUGGESTION_INTERVAL,
        help="Seconds between streaming frames",
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, suggestion_interval=args.suggestion_interval)
    asyncio.run(main.run())