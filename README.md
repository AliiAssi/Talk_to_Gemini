# Gemini Live Audio & Video Streaming Demo

This is a prototype that streams audio and video (from webcam or screen), sends it to **Google Gemini 2.0 Flash (Live API)**, and receives real-time AI-generated audio responses.

> Voice + Vision + Gemini in real-time.

---

## 🔧 Setup

Install required dependencies:

```bash
pip install google-genai opencv-python pyaudio pillow mss
```

---

## 🚀 Run the App

```bash
python main.py --mode camera
```

Available modes:
- `camera`: Use webcam
- `screen`: Share screen
- `none`: No video

---

## 📚 How It Works

- Captures **audio** using `pyaudio` and sends it to the Gemini model.
- Captures **video frames** (from webcam or screen) and sends them too.
- Plays back **Gemini's audio response** in real-time.
- Supports **live conversation** through the terminal.

---

## 🌐 Gemini Live API

This demo uses:
- `models/gemini-2.0-flash-live-001`
- Audio responses via `Puck` voice
- LiveConnect stream with support for audio + image input

Quickstart reference: [Google Gemini Cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py)

---

## ⚠️ Notes

- This is an **experimental prototype**.
- Requires a valid **Google API key** to work.

---

## 🙋‍♂️ Why?

This project is built to explore the **real-time capabilities of Gemini** with multimodal input (audio + vision), and pave the way for interactive AI interfaces.
