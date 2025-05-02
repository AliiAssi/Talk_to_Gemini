import pyaudio

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