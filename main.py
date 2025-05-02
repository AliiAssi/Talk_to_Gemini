import argparse
import asyncio
from main_app import GeminiLiveApp

def main():
    parser = argparse.ArgumentParser(description="Gemini Live API Application")
    parser.add_argument(
        "--mode",
        type=str,
        default="screen",
        choices=["camera", "screen", "none"],
        help="Source for visual input (camera, screen, or none)",
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Time interval in seconds between proactive suggestions",
    )
    args = parser.parse_args()

    app = GeminiLiveApp(video_mode=args.mode, suggestion_interval=args.interval)
    asyncio.run(app.run())

if __name__ == "__main__":
    main()