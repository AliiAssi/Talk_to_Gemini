import asyncio
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
import pyaudio

pya = pyaudio.PyAudio()
"""
Captures and plays back audio via microphone and speaker.
Usage:
```python
from audio_handler import AudioHandler
handler = AudioHandler()
await handler.setup(session, audio_in_queue, out_queue)
# To listen from mic:
await handler.listen_audio()
# To receive from API:
await handler.receive_audio()
# To play back:
await handler.play_audio()
handler.cleanup()
```"""
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
