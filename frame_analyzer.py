import asyncio
from config import DEFAULT_SUGGESTION_INTERVAL
import time
import io
import base64

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
