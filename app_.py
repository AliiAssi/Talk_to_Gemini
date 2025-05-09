import eventlet

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import io
import traceback
import time
from dotenv import load_dotenv
import PIL.Image

from Gemini_Communication.gemini_client import GeminiClient

load_dotenv()


app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'screen-analyzer-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60)

# Global client instance
gemini_client = GeminiClient()

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
    gemini_client.stop_session(sid)

@socketio.on('start_session')
def handle_start_session():
    """Start a new analysis session"""
    session_id = request.sid
    if gemini_client.is_session_active(session_id):
        emit('status', {'message': 'Session already running'})
        return
    
    print(f'Starting new session for {session_id}')
    
    # Start the analysis session in a background task
    socketio.start_background_task(
        target=gemini_client.start_analysis_session,
        session_id=session_id,
        socketio=socketio
    )
    
    emit('status', {'message': 'Session started'})
    # Test message to verify communication
    emit('text_response', {'text': 'Session started, ready to analyze your screen'})

@socketio.on('stop_session')
def handle_stop_session():
    """Stop the current analysis session"""
    gemini_client.stop_session(request.sid)
    emit('status', {'message': 'Session stopped'})

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


@socketio.on('screen_data')
def handle_screen_data(data):
    """Process incoming screen capture data"""
    sid = request.sid
    if not gemini_client.is_session_active(sid):
        return
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        img = PIL.Image.open(io.BytesIO(image_data))
        
        # Resize if needed
        img.thumbnail([1024, 1024])
        gemini_client.set_frame_for_analysis(sid, img)
    except Exception as e:
        print(f"Error processing screen data: {e}")
        traceback.print_exc()

@socketio.on('audio_data')
def handle_audio_data(data):
    """Process incoming audio data"""
    sid = request.sid
    if not gemini_client.has_session(sid):
        return
    print(f"Processing incoming audio data from {sid}")

    try:
        # Extract base64 data correctly
        audio_data = base64.b64decode(data['audio'].split(',')[1] if ',' in data['audio'] else data['audio'])
        
        # Send audio data to Gemini with proper MIME type including sample rate
        gemini_client.send_audio_data(sid, audio_data, data.get('sampleRate', 16000))
    except Exception as e:
        print(f"Error processing audio data: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)