import eventlet

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import asyncio
import base64
import io
import traceback
import time
import os
import threading
from dotenv import load_dotenv

import PIL.Image
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
API_KEY = "AIzaSyCex6BkIKAtPP3WpMHBcw0PvskrmUAwrhw"  # Better to use environment variable

# Configuration
MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_SUGGESTION_INTERVAL = 20

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'screen-analyzer-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60)

# Initialize Google Genai client
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

# Global data store
class SharedState:
    def __init__(self):
        self.frame_for_analysis = None
        self.last_suggestion_time = 0
        self.session = None
        self.active_sessions = {}
        self.loop = None
        self.session_tasks = {}
        self.response_queues = {}  # Add queue for each session

shared_state = SharedState()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle new client connections"""
    sid = request.sid
    print(f'Client connected: {sid}')
    # Room is automatically created for sid
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections"""
    sid = request.sid
    print(f'Client disconnected: {sid}')
    if sid in shared_state.active_sessions:
        stop_session(sid)

@socketio.on('start_session')
def handle_start_session():
    """Start a new analysis session"""
    session_id = request.sid
    if session_id in shared_state.active_sessions:
        emit('status', {'message': 'Session already running'})
        return
    
    print(f'Starting new session for {session_id}')
    shared_state.active_sessions[session_id] = {
        'running': True,
        'last_suggestion_time': time.time()
    }
    
    # Add a response queue for this session
    shared_state.response_queues[session_id] = []
    
    # Start the response processor for this client
    socketio.start_background_task(process_response_queue, session_id)
    
    # Start the analysis session in a background task
    socketio.start_background_task(
        target=start_analysis_session,
        session_id=session_id
    )
    
    emit('status', {'message': 'Session started'})
    # Test message to verify communication
    emit('text_response', {'text': 'Session started, ready to analyze your screen'})

@socketio.on('stop_session')
def handle_stop_session():
    """Stop the current analysis session"""
    stop_session(request.sid)
    emit('status', {'message': 'Session stopped'})

# Add these routes to your Flask app for testing Socket.IO communication

@socketio.on('test_connection')
def handle_test_connection(data):
    """Handle test connection ping from client"""
    sid = request.sid
    print(f'Test connection from {sid}')
    
    # Return acknowledgment
    return {'status': 'ok', 'sid': sid}

@socketio.on('test_message')
def handle_test_message(data):
    """Handle test message from client"""
    sid = request.sid
    print(f'Test message from {sid}: {data}')
    
    # Send back a test response
    emit('text_response', {'text': f'Received test message: {data.get("message", "")}'})

# Add a testing route to check server status
@app.route('/api/status')
def api_status():
    """API status endpoint"""
    active_sessions = list(shared_state.active_sessions.keys())
    return {
        'status': 'running',
        'active_sessions': active_sessions,
        'session_count': len(active_sessions)
    }

# Add a debug route to send test messages
@app.route('/api/debug/<string:session_id>')
def api_debug(session_id):
    """Send a test message to a specific session"""
    if session_id in shared_state.active_sessions:
        socketio.emit('text_response', {'text': 'This is a debug test message from the server'}, room=session_id)
        return {'success': True, 'message': f'Test message sent to {session_id}'}
    else:
        return {'success': False, 'message': f'No active session with ID: {session_id}'}, 400

def stop_session(session_id):
    """Stop an active session and clean up resources"""
    if session_id in shared_state.active_sessions:
        print(f'Stopping session for {session_id}')
        shared_state.active_sessions[session_id]['running'] = False
        if session_id in shared_state.session_tasks:
            for task in shared_state.session_tasks[session_id]:
                task.cancel()
            del shared_state.session_tasks[session_id]
        del shared_state.active_sessions[session_id]
        if session_id in shared_state.response_queues:
            del shared_state.response_queues[session_id]

@socketio.on('screen_data')
def handle_screen_data(data):
    """Process incoming screen capture data"""
    sid = request.sid
    if sid not in shared_state.active_sessions:
        return
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        img = PIL.Image.open(io.BytesIO(image_data))
        
        # Resize if needed
        img.thumbnail([1024, 1024])
        shared_state.active_sessions[sid]['frame_for_analysis'] = img
    except Exception as e:
        print(f"Error processing screen data: {e}")
        traceback.print_exc()

@socketio.on('audio_data')
def handle_audio_data(data):
    """Process incoming audio data"""
    sid = request.sid
    if sid not in shared_state.active_sessions or 'session' not in shared_state.active_sessions[sid]:
        return
    print(f"Processing incoming audio data from {sid}")

    try:
        session = shared_state.active_sessions[sid]['session']
        # Extract base64 data correctly
        audio_data = base64.b64decode(data['audio'].split(',')[1] if ',' in data['audio'] else data['audio'])
        
        # Send audio data to Gemini with proper MIME type including sample rate
        asyncio.run_coroutine_threadsafe(
            session.send_realtime_input(
                media=types.Blob(data=audio_data, mime_type=f"audio/pcm;rate={data.get('sampleRate', 16000)}")
            ),
            shared_state.loop
        )
    except Exception as e:
        print(f"Error processing audio data: {e}")
        traceback.print_exc()

async def send_system_prompt(session):
    """Send initial system prompt to Gemini"""
    system_prompt = """You are an intelligent screen analyzer and game assistant that provides helpful, 
    contextual suggestions based on what you see on the user's screen.
    
    Focus on:
    1. Identifying what the user is doing.
    2. Suggesting useful shortcuts, tips, or improvements.
    3. Being concise and non-intrusive.
    4. Minimal few simple words.
    5. Adapting to context - only mention significant changes.
    6. Being helpful without being annoying.
    
    Keep suggestions brief and actionable."""
    print(f"Sending the system prompt")
    await session.send_realtime_input(text=system_prompt)

# Queue the responses to be processed in the SocketIO thread
def queue_response(session_id, event_type, data):
    """Add a response to the queue for a specific session"""
    if session_id in shared_state.response_queues:
        shared_state.response_queues[session_id].append((event_type, data))
        return True
    return False

# Process the queue in the SocketIO thread context
def process_response_queue(session_id):
    """Process queued responses for a session"""
    print(f"Starting response queue processor for {session_id}")
    while session_id in shared_state.active_sessions and session_id in shared_state.response_queues:
        if shared_state.response_queues[session_id]:
            try:
                event_type, data = shared_state.response_queues[session_id].pop(0)
                print(f"Emitting {event_type} for {session_id}")
                socketio.emit(event_type, data, room=session_id)
            except Exception as e:
                print(f"Error processing response queue: {e}")
                traceback.print_exc()
        eventlet.sleep(0.01)  # Short sleep to prevent CPU hogging

async def process_gemini_responses(session, session_id):
    """Process responses from Gemini"""
    while session_id in shared_state.active_sessions and shared_state.active_sessions[session_id]['running']:
        try:
            turn = session.receive()
            
            async for response in turn:
                if session_id not in shared_state.active_sessions:
                    print(f"[{session_id}] ⚠️ Session dropped mid-response, breaking out")
                    break

                # If it's audio data
                if data := response.data:
                    try:
                        if len(data) > 0:
                            wav_header = create_wav_header(len(data), channels=1, sample_rate=24000, bits_per_sample=16)
                            full_audio = bytearray(wav_header) + bytearray(data)
                            audio_b64 = base64.b64encode(full_audio).decode('utf-8')
                            
                            # Queue the response instead of direct emit
                            queue_response(session_id, 'audio_response', {'audio': audio_b64})
                    except Exception as e:
                        print(f"[{session_id}] 🚨 Error encoding audio: {e}")
                        traceback.print_exc()
                elif text := response.text:
                    # Queue the text response
                    queue_response(session_id, 'text_response', {'text': text})

            print(f"[{session_id}] 🔚 Finished draining this turn")
        except asyncio.CancelledError:
            print(f"[{session_id}] ❌ process_gemini_responses task cancelled")
            break
        except Exception as e:
            print(f"[{session_id}] 🚨 Error processing Gemini responses: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

def create_wav_header(data_size, channels=1, sample_rate=24000, bits_per_sample=16):
    """Create a proper WAV header for raw PCM data"""
    header = bytearray(44)
    
    # RIFF identifier
    header[0:4] = b'RIFF'
    # RIFF chunk length
    header[4:8] = (36 + data_size).to_bytes(4, byteorder='little')
    # WAVE identifier
    header[8:12] = b'WAVE'
    # fmt chunk identifier
    header[12:16] = b'fmt '
    # fmt chunk length
    header[16:20] = (16).to_bytes(4, byteorder='little')
    # audio format (PCM)
    header[20:22] = (1).to_bytes(2, byteorder='little')
    # number of channels
    header[22:24] = (channels).to_bytes(2, byteorder='little')
    # sample rate
    header[24:28] = (sample_rate).to_bytes(4, byteorder='little')
    # byte rate (sample rate * block align)
    header[28:32] = (sample_rate * channels * bits_per_sample // 8).to_bytes(4, byteorder='little')
    # block align (channels * bits per sample / 8)
    header[32:34] = (channels * bits_per_sample // 8).to_bytes(2, byteorder='little')
    # bits per sample
    header[34:36] = (bits_per_sample).to_bytes(2, byteorder='little')
    # data chunk identifier
    header[36:40] = b'data'
    # data chunk length
    header[40:44] = (data_size).to_bytes(4, byteorder='little')
    
    return header

async def smart_analysis(session_id):
    """Periodically analyze screen content and provide suggestions"""
    session = shared_state.active_sessions[session_id]['session']
    print(f"Starting smart analysis for {session_id}")
    while session_id in shared_state.active_sessions and shared_state.active_sessions[session_id]['running']:
        try:
            now = time.time()
            session_data = shared_state.active_sessions[session_id]
            
            if 'frame_for_analysis' in session_data and now - session_data.get('last_suggestion_time', 0) >= DEFAULT_SUGGESTION_INTERVAL:
                # Prepare and send the current frame for analysis
                frame = session_data['frame_for_analysis']
                buf = io.BytesIO()
                frame.save(buf, format="jpeg")
                raw = buf.getvalue()
                blob = types.Blob(data=raw, mime_type="image/jpeg")
                
                # Create a context-aware prompt
                prompt = """Analyze this screen and provide one brief, helpful suggestion.
                Be specific about what you see and keep your response under 3 sentences.
                Only mention significant changes or opportunities for improvement."""
                
                print(f"[{session_id}] Analyzing screen content...")
                queue_response(session_id, 'status', {'message': 'Analyzing screen...'})
                
                # Send the analysis prompt
                await session.send_realtime_input(text=prompt)
                
                # Send the current frame for analysis
                await session.send_realtime_input(media=blob)
                
                session_data['last_suggestion_time'] = now
                
            await asyncio.sleep(3)  # Check more frequently than the suggestion interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in smart analysis: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

def initialize_event_loop():
    """Create and start an event loop in a background thread"""
    def run_loop_forever(loop):
        asyncio.set_event_loop(loop)
        print("Event loop started in background thread")
        try:
            loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
            traceback.print_exc()
    
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_loop_forever, args=(loop,), daemon=True)
    thread.start()
    return loop

def start_analysis_session(session_id):
    """Set up and run the analysis session"""
    if not shared_state.loop:
        print("Creating new event loop")
        shared_state.loop = initialize_event_loop() 
    loop = shared_state.loop
    
    async def run_session():
        try:
            print(f"[{session_id}] Attempting to connect to Gemini model...")
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                print(f"[{session_id}] Successfully connected to Gemini model")
                shared_state.active_sessions[session_id]['session'] = session

                # send your system prompt
                print(f"[{session_id}] Sending system prompt...")
                await send_system_prompt(session)
                print(f"[{session_id}] System prompt sent")

                # Add a test response to confirm communication
                queue_response(session_id, 'text_response', {'text': 'Connected to Gemini. Ready to analyze.'})

                # spin up your two tasks under the same context
                print(f"[{session_id}] Creating response and analysis tasks...")
                response_task = asyncio.create_task(process_gemini_responses(session, session_id))
                analysis_task = asyncio.create_task(smart_analysis(session_id))
                shared_state.session_tasks[session_id] = [response_task, analysis_task]
                print(f"[{session_id}] Tasks created successfully")

                # wait for them to finish (or be cancelled)
                await asyncio.gather(response_task, analysis_task)
        except Exception as e:
            print(f"Session error: {e}")
            traceback.print_exc()
            queue_response(session_id, 'status', {'message': f'Error: {str(e)}'})

    asyncio.run_coroutine_threadsafe(run_session(), loop)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)