from google import genai
from google.genai import types

# Initialize the Gemini API client
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key="YOUR_API_KEY_HERE"  # Replace with your actual API key
)

# Live audio-response configuration
CONFIG = types.LiveConnectConfig(
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)