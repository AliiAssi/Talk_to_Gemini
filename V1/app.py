import asyncio
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from test import ScreenAnalyzer, client, CONFIG, MODEL, DEFAULT_MODE, DEFAULT_SUGGESTION_INTERVAL

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

class WebScreenAnalyzer(ScreenAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False
        self.websocket = None

    async def run_async(self):
        while True:
            if self.active:
                try:
                    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                        self.session = session
                        self.audio_in_queue = asyncio.Queue()
                        self.out_queue = asyncio.Queue(maxsize=5)

                        tasks = [
                            self.send_realtime(),
                            self.listen_audio(),
                            self.get_screen(),
                            self.receive_audio(),
                            self.play_audio(),
                            self.smart_analysis()
                        ]
                        
                        await asyncio.gather(*tasks)
                        
                except Exception as e:
                    print(f"Analyzer error: {str(e)}")
                    self.active = False
            await asyncio.sleep(1)

def run_analyzer(analyzer):
    asyncio.run(analyzer.run_async())

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_analysis')
def handle_start_analysis():
    analyzer = WebScreenAnalyzer(video_mode=DEFAULT_MODE, suggestion_interval=DEFAULT_SUGGESTION_INTERVAL)
    analyzer.active = True
    socketio.start_background_task(target=run_analyzer, analyzer=analyzer)
    emit('analysis_started', {'status': 'ready'})

if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True)