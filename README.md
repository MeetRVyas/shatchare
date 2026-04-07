# LLM Chat Rooms

A minimal self-hosted chat server with private rooms, shared LLM config, and Google Gemini support.

## Features

- **Private Rooms** — 6-character codes, sharable via link
- **Real-time sync** — all room members see messages and config changes instantly via WebSocket
- **Multiple API keys** — add several Gemini keys; rate-limited ones are removed automatically
- **Model selection** — Gemini 2.0 Flash Thinking Preview or Flash Lite Preview
- **Shared config** — model, system prompt, temperature, and token limit are room-wide
- **Concurrent requests** — async semaphore prevents hammering the API

## Setup

```bash
cd app
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

## Usage

1. **Create a room** — enter your name, click Create Room
2. **Open Settings** (top right) → add one or more Gemini API keys
3. **Select model**, adjust system prompt and generation settings
4. **Share the room link** or 6-character code with others
5. **Chat** — all users in the room share the conversation and config

## API Key Rotation

- Add multiple keys for higher throughput
- If a key hits rate limits (429 / RESOURCE_EXHAUSTED), it is automatically removed and the next key is tried
- A notice is broadcast to all room members when keys are removed
- You can add new keys at any time

## Models

| ID | Display Name |
|----|-------------|
| `gemini-2.0-flash-thinking-exp` | Gemini 2.0 Flash Thinking Preview |
| `gemini-2.0-flash-lite` | Gemini 2.0 Flash Lite Preview |

## Architecture

```
Browser ──WS──▶ FastAPI (main.py)
                  │
                  ├─ rooms{}          in-memory room state
                  ├─ room_connections{} active WebSocket sets
                  ├─ LLM_SEMAPHORE   caps concurrent Gemini calls
                  └─ handle_llm()    async task per message
                       └─ call_gemini() rotates keys on rate limit
```
