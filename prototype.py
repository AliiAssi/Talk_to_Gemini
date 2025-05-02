"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

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

FORMAT = pyaudio.paInt16 # 
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024 #audio processing variable

MODEL = "models/gemini-2.0-flash-live-001"


DEFAULT_MODE = "screen"

# Time interval in seconds for proactive suggestions
DEFAULT_SUGGESTION_INTERVAL = 30 #seconds

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key="AIzaSyCex6BkIKAtPP3WpMHBcw0PvskrmUAwrhw"
)


# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
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

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        # Store frame for analysis
        self.frame_for_analysis = img.copy()

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Store frame for analysis
        self.frame_for_analysis = img.copy()

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

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
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
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
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def periodic_suggestions(self):
        """Periodically analyze frames and provide proactive suggestions."""
        while True:
            current_time = time.time()
            # Check if it's time for a suggestion
            if current_time - self.last_suggestion_time >= self.suggestion_interval and self.frame_for_analysis is not None:
                # Create a prompt for frame analysis
                prompt = "Based on what you see in this frame, can you provide a helpful suggestion or observation? Be concise and natural."
                
                # Prepare the frame for sending
                image_io = io.BytesIO()
                self.frame_for_analysis.save(image_io, format="jpeg")
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

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Add the periodic suggestions task
                tg.create_task(self.periodic_suggestions())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--suggestion-interval",
        type=int,
        default=DEFAULT_SUGGESTION_INTERVAL,
        help="Time interval in seconds between proactive suggestions",
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, suggestion_interval=args.suggestion_interval)
    asyncio.run(main.run())