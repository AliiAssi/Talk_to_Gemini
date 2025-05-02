"""
Handles CLI-based text input and sending to Gemini API.
Usage:
```python
from text_handler import TextHandler
th = TextHandler()
await th.setup(session)
await th.send_text()  # Blocks until user types 'q'
```"""
import asyncio


class TextHandler:
    """
    Handles text input from the user
    """
    def __init__(self):
        self.session = None
    
    async def setup(self, session):
        """Set up the text handler"""
        self.session = session
    
    async def send_text(self):
        """Get text input from the user and send to the Gemini API"""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            # Exit if the user types 'q'
            if text.lower() == "q":
                break
            # Send the text to the API
            await self.session.send(input=text or ".", end_of_turn=True)
        
        # Signal completion
        return True
